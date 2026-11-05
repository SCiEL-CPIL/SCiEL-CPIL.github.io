import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    prescribed = [1, 2, 5, 11]
    prescribed_set = frozenset(prescribed)
    best_size = 0
    best_set = set(prescribed_set)

    # Pre-compute sums for prescribed set
    def get_sums(s):
        sums = set()
        lst = sorted(s)
        for i in range(len(lst)):
            for j in range(i, len(lst)):
                sums.add(lst[i] + lst[j])
        return sums

    initial_sums = get_sums(prescribed_set)

    # Helper to check if x can be added to a set with given sums
    def can_add(x, s_set, s_sums):
        if x in s_set:
            return False
        # Check x + y for all y in s_set
        for y in s_set:
            if x + y in s_sums:
                return False
        # Check 2x
        if 2 * x in s_sums:
            return False
        return True

    # Helper to update sums when adding x
    def add_to_sums(x, s_set, s_sums):
        new_sums = set(s_sums)
        for y in s_set:
            new_sums.add(x + y)
        new_sums.add(2 * x)
        return new_sums

    def greedy_fill(current_set, current_sums, candidates):
        """Greedily add elements from candidates to current_set."""
        for x in candidates:
            if x in current_set:
                continue
            if can_add(x, current_set, current_sums):
                current_set.add(x)
                current_sums = add_to_sums(x, current_set, current_sums)
        return current_set, current_sums

    def build_conflict_graph():
        """
        Build a conflict graph where vertices are integers 1..284.
        Two vertices x, y conflict if there exist (possibly equal) pairs
        (a,b) and (c,d) with a<=b, c<=d, a+b = c+d, and {x,y} intersects
        the union of these pairs in a way that creates a duplicate sum.
        
        More precisely: x and y are incompatible if there exist a,b in the
        current set such that x+y conflicts, OR if adding both x and y
        would create a duplicate sum among themselves or with existing elements.
        
        For the static graph (independent of current set), two candidates x,y
        conflict if:
        - x == y (can't pick same element twice) - handled separately
        - There exist a,b such that a+b = x+y where {a,b} != {x,y}
          This is too broad. Let's think differently.
        
        Actually, for a Sidon set, we need all pairwise sums a+b (a<=b) to be distinct.
        So x and y can coexist if:
        1. x != y
        2. For any other pair (a,b) in the set, a+b != x+y (unless {a,b} == {x,y})
        3. 2x != 2y (i.e., x != y, already covered)
        4. x+y != 2z for any z in the set (z != x, z != y)
        5. x+y != a+b for any other pair (a,b)
        
        The conflict between x and y depends on the rest of the set, so a static
        graph doesn't fully capture it. Instead, let's use a different approach:
        precompute for each pair (x,y) whether they "conflict" in the sense that
        there exist a,b (not both x and y) such that a+b = x+y.
        
        Actually, the key constraint is: if we have elements {..., x, y, ...}, then
        x+y must not equal any other pair sum. So x and y conflict if there exist
        distinct a,b (where {a,b} != {x,y}) such that a+b = x+y.
        
        But this is still set-dependent. Let me use a simpler approach:
        Precompute for each value s (sum), which pairs (a,b) with a<=b give a+b=s.
        Then x and y are "sum-conflicting" if there's another pair giving x+y.
        """
        # For each sum s, store all pairs (a,b) with a<=b, a+b=s
        sum_pairs = {}
        for a in range(1, 285):
            for b in range(a, 285):
                s = a + b
                if s not in sum_pairs:
                    sum_pairs[s] = []
                sum_pairs[s].append((a, b))
        
        # Two elements x,y (x<y) are in conflict if there exists a pair (a,b) 
        # with a<=b, (a,b) != (x,y), such that a+b = x+y
        # AND at least one of a,b is not x or y (otherwise it's the same pair)
        # Actually, if (a,b) = (x,y) it's the same, so we need (a,b) != (x,y)
        # and a+b = x+y. Since x+y is fixed, we just need another pair.
        
        conflict = [set() for _ in range(285)]
        for x in range(1, 285):
            for y in range(x + 1, 285):
                s = x + y
                pairs = sum_pairs.get(s, [])
                # Check if there's another pair
                for a, b in pairs:
                    if (a, b) != (x, y):
                        conflict[x].add(y)
                        conflict[y].add(x)
                        break
        
        # Also, x and x (double) conflicts: 2x must be unique
        # 2x conflicts with a+b where a+b=2x and {a,b} != {x,x}
        for x in range(1, 285):
            s = 2 * x
            pairs = sum_pairs.get(s, [])
            for a, b in pairs:
                if not (a == x and b == x):
                    # 2x conflicts with a+b, meaning x can't coexist with both a and b
                    # if a != b, then x conflicts with... hmm, this is more complex
                    # Actually, if a+b = 2x and {a,b} != {x,x}, then having x in the set
                    # means 2x is taken, so a and b can't both be in the set.
                    # This is a 3-way constraint, not pairwise.
                    pass
        
        return conflict, sum_pairs

    def mis_bit_parallel(vertices, conflict_bits, time_limit):
        """
        Bit-parallel Maximum Independent Set using Bron-Kerbosch with pivot.
        vertices: set of vertex indices (1-indexed, using 0-indexed internally)
        conflict_bits: list of bitsets, conflict_bits[i] has bit j set if i conflicts with j
        Returns the largest independent set found.
        """
        # Convert to 0-indexed
        v_list = sorted(vertices)
        v_to_idx = {v: i for i, v in enumerate(v_list)}
        n = len(v_list)
        
        if n == 0:
            return set()
        
        # Build adjacency bitsets for 0-indexed vertices
        adj = [0] * n
        for i in range(n):
            vi = v_list[i]
            for j in range(i + 1, n):
                vj = v_list[j]
                if j in conflict_bits[vi]:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        best_size = 0
        best_set = set()
        start = time.time()
        
        def bron_kerbosch(R, P, X):
            nonlocal best_size, best_set
            if time.time() - start > time_limit:
                return
            
            if len(P) == 0 and len(X) == 0:
                if len(R) > best_size:
                    best_size = len(R)
                    best_set = set(R)
                return
            
            # Pivot selection: choose u from P ∪ X with max degree in P
            union_PX = P | X
            if not union_PX:
                if len(R) > best_size:
                    best_size = len(R)
                    best_set = set(R)
                return
            
            # Find pivot with max neighbors in P
            best_pivot = -1
            best_deg = -1
            for u in union_PX:
                deg = bin(P & adj[u]).count('1')
                if deg > best_deg:
                    best_deg = deg
                    best_pivot = u
            
            # Candidates: P \\ N(pivot)
            candidates = P & ~adj[best_pivot]
            
            for v in list(candidates):
                # Add v to R, remove from P, add to X
                new_R = R | {v}
                new_P = P & ~adj[v] & ~(1 << v)
                new_X = X & ~adj[v] & ~(1 << v)
                P &= ~(1 << v)
                X |= (1 << v)
                
                bron_kerbosch(new_R, new_P, new_X)
                
                # Backtrack
                P |= (1 << v)
                X &= ~(1 << v)
                
                if len(R) + len(P) <= best_size:
                    return
        
        # Initial call
        P_all = (1 << n) - 1
        bron_kerbosch(set(), P_all, set())
        
        return best_set

    def kick_search(current_set, current_sums, time_limit=5.0):
        """
        Perform kick-based local search.
        Removes k random elements and greedily refills.
        """
        best_local_set = set(current_set)
        best_local_sums = current_sums
        start = time.time()

        all_candidates = list(range(1, 285))

        while time.time() - start < time_limit:
            # Decide kick size: 1 or 2
            kick_size = 1 if len(current_set) < 15 else random.choice([1, 2])

            removable = [x for x in current_set if x not in prescribed_set]
            if len(removable) < kick_size:
                break

            # Select elements to remove
            to_remove = random.sample(removable, kick_size)

            # Remove them
            temp_set = set(current_set)
            for x in to_remove:
                temp_set.remove(x)
            temp_sums = get_sums(temp_set)

            # Greedily add back from all candidates not in temp_set
            # Shuffle candidates to randomize the fill order
            candidates = [x for x in all_candidates if x not in temp_set]
            random.shuffle(candidates)

            temp_set, temp_sums = greedy_fill(temp_set, temp_sums, candidates)

            # Update local best
            if len(temp_set) > len(best_local_set):
                best_local_set = set(temp_set)
                best_local_sums = temp_sums
                current_set = set(temp_set)
                current_sums = temp_sums
            elif len(temp_set) == len(best_local_set) and random.random() < 0.3:
                # Accept same-size solution with some probability to explore
                current_set = set(temp_set)
                current_sums = temp_sums

        return best_local_set, best_local_sums

    def swap_local_search(current_set, current_sums, time_limit=5.0):
        """
        Swap-based local search: try removing one element and adding another.
        """
        best_set = set(current_set)
        best_sums = current_sums
        start = time.time()
        
        while time.time() - start < time_limit:
            improved = False
            removable = [x for x in best_set if x not in prescribed_set]
            random.shuffle(removable)
            
            for x_remove in removable:
                if time.time() - start >= time_limit:
                    break
                
                temp_set = set(best_set)
                temp_set.remove(x_remove)
                temp_sums = get_sums(temp_set)
                
                # Try all possible additions
                for y_add in range(1, 285):
                    if y_add in temp_set:
                        continue
                    if can_add(y_add, temp_set, temp_sums):
                        # This swap maintains the same size, but may enable future improvements
                        # Accept with some probability to explore
                        if random.random() < 0.1:
                            best_set = temp_set
                            best_set.add(y_add)
                            best_sums = get_sums(best_set)
                            improved = True
                            break
                if improved:
                    break
            
            if not improved and len(removable) == 0:
                break
        
        return best_set, best_sums

    def csp_backtracking(selected, used_sums, candidates, target_size, time_limit):
        """
        CSP-style backtracking with MRV heuristic.
        """
        start = time.time()
        best_found = set(selected)
        best_sums_found = set(used_sums)

        def backtrack(sel, sums, cand_list, depth):
            nonlocal best_found, best_sums_found
            if time.time() - start > time_limit:
                return

            if len(sel) >= target_size:
                if len(sel) > len(best_found):
                    best_found = set(sel)
                    best_sums_found = set(sums)
                return

            remaining_needed = target_size - len(sel)
            if len(cand_list) < remaining_needed:
                return

            # Sort candidates by how many conflicts they have with current sel
            scored = []
            for x in cand_list:
                if x in sel:
                    continue
                conflicts = 0
                for y in sel:
                    if x + y in sums:
                        conflicts += 1
                if 2 * x in sums:
                    conflicts += 1
                scored.append((conflicts, x))

            scored.sort(key=lambda t: t[0])

            max_branch = min(50, len(scored))
            for _, x in scored[:max_branch]:
                if time.time() - start > time_limit:
                    return

                valid = True
                for y in sel:
                    if x + y in sums:
                        valid = False
                        break
                if not valid:
                    continue
                if 2 * x in sums:
                    continue

                new_sel = set(sel)
                new_sel.add(x)
                new_sums = set(sums)
                for y in sel:
                    new_sums.add(x + y)
                new_sums.add(2 * x)

                new_cands = [c for c in cand_list if c != x]

                backtrack(new_sel, new_sums, new_cands, depth + 1)

                if len(best_found) >= target_size:
                    return

        backtrack(set(selected), set(used_sums), list(candidates), 0)
        return best_found, best_sums_found

    # Main search loop
    num_restarts = 0
    while time.time() < deadline - 10:
        num_restarts += 1

        # 1. Random Greedy Construction
        current_set = set(prescribed_set)
        current_sums = get_sums(current_set)

        candidates_list = list(range(12, 285))
        random.shuffle(candidates_list)

        current_set, current_sums = greedy_fill(current_set, current_sums, candidates_list)

        # 2. CSP Backtracking to try to reach target size
        remaining = deadline - time.time()
        if remaining > 5:
            target = max(len(current_set) + 1, 18)
            csp_time = min(3.0, max(0.5, remaining * 0.05))
            csp_candidates = [x for x in range(1, 285) if x not in current_set]
            random.shuffle(csp_candidates)
            csp_set, csp_sums = csp_backtracking(
                current_set, current_sums, csp_candidates, target, csp_time
            )
            if len(csp_set) > len(current_set):
                current_set = csp_set
                current_sums = csp_sums

        # 3. Kick-based Local Search
        remaining = deadline - time.time()
        if remaining > 3:
            kick_time = min(5.0, max(0.5, remaining * 0.1))
            current_set, current_sums = kick_search(current_set, current_sums, kick_time)

        # 4. Swap-based Local Search
        remaining = deadline - time.time()
        if remaining > 2:
            swap_time = min(3.0, max(0.3, remaining * 0.08))
            current_set, current_sums = swap_local_search(current_set, current_sums, swap_time)

        # 5. Update Best
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)

            if best_size >= 20:
                break

    # Final polish: try all single swaps on the best set
    if time.time() < deadline - 2:
        best_sums = get_sums(best_set)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            removable = [x for x in best_set if x not in prescribed_set]
            for x_remove in removable:
                if time.time() >= deadline - 2:
                    break
                temp_set = set(best_set)
                temp_set.remove(x_remove)
                temp_sums = get_sums(temp_set)

                for y_add in range(1, 285):
                    if y_add in temp_set:
                        continue
                    if can_add(y_add, temp_set, temp_sums):
                        new_set = set(temp_set)
                        new_set.add(y_add)
                        if len(new_set) > len(best_set):
                            best_set = new_set
                            best_sums = get_sums(best_set)
                            improved = True
                            break
                if improved:
                    break

    # Final validation before returning
    final_sums = get_sums(best_set)
    if len(final_sums) != len(best_set) * (len(best_set) + 1) // 2:
        return sorted(prescribed)

    return sorted(list(best_set))
