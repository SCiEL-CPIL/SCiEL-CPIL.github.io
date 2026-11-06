import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 55.0
    max_val = 4096

    # Phase 1: Fast Greedy to get a baseline
    def greedy():
        S = []
        used_sums = set()
        for x in range(1, max_val + 1):
            k = len(S)
            # Check 3x
            s1 = 3 * x
            if s1 in used_sums:
                continue
            # Check 2x + s
            valid = True
            for s in S:
                if (2 * x + s) in used_sums:
                    valid = False
                    break
            if not valid:
                continue
            # Check x + si + sj
            for i in range(k):
                for j in range(i, k):
                    if (x + S[i] + S[j]) in used_sums:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                S.append(x)
                used_sums.add(3 * x)
                for s in S[:-1]:
                    used_sums.add(2 * x + s)
                for i in range(k):
                    for j in range(i, k):
                        used_sums.add(x + S[i] + S[j])
        return S

    best_set = greedy()
    best_len = len(best_set)
    
    # Phase 2: Bitset-based Maximum Independent Set (MIS) on the conflict graph
    # The conflict graph has nodes 1..4096. An edge exists between i and j if adding both to a set
    # (or having them in the set) causes a triple sum collision.
    # Actually, the constraint is: for any set S, all unordered triples (i,j,k) with i<=j<=k must have distinct sums.
    # This is equivalent to: no two distinct triples (i,j,k) and (i',j',k') (where i<=j<=k, i'<=j'<=k') have the same sum.
    # This is NOT a simple pairwise constraint between elements. It's a global constraint on the set.
    # However, we can model this as a graph problem if we define conflicts between *triples*? No, that's huge.
    #
    # Wait, the directive says: "Replace the incremental collision checking with a precomputed pairwise conflict matrix C[i][j] derived from all possible triple sum collisions... solve the resulting maximum independent set problem".
    # This implies that the problem *can* be reduced to a pairwise conflict graph.
    # Let's re-read the B_3 definition. A B_3 set requires all sums of 3 elements to be distinct.
    # If we pick a set S, we require that for any i,j,k in S (with repetition allowed), the sum is unique.
    # This is a 3-wise constraint. It is NOT reducible to a pairwise conflict graph in the standard sense (where edges mean "cannot be together").
    # Unless... the "conflict" is defined differently?
    #
    # Let's look at the structure.
    # If i+j+k = i'+j'+k', and the multisets {i,j,k} != {i',j',k'}, then we have a conflict.
    # This is a 3-uniform hypergraph constraint.
    # The directive suggests a "pairwise conflict matrix". This might be a misinterpretation of the problem structure by the prompt generator, OR there is a specific way to project this.
    #
    # However, we can still use a "local search with bitsets" approach which is structurally different from the current incremental check.
    # The current program uses `can_add` which is O(N^2) per insertion.
    #
    # Let's try a different structural approach: **Simulated Annealing / Randomized Local Search with a different neighborhood**.
    # The current program does greedy + local search (add/swap).
    # The inspiration mentions "Bianchi" and "B_3".
    #
    # Let's implement a faster local search using NumPy for vectorized checks where possible, or a smarter pruning.
    #
    # Actually, let's look at the "pairwise conflict" idea again.
    # If we fix two elements i and j, the sum i+j+k must be unique.
    # This means for any k, i+j+k is not equal to any other triple sum.
    #
    # Let's stick to a robust, fast local search but with a **different representation**:
    # Instead of maintaining a set of sums, we can maintain a boolean array of used sums (size 3*4096 = 12288).
    # This is faster for membership testing than a Python set of ints? Maybe not significantly, but array indexing is fast.
    #
    # Let's try **Simulated Annealing** on the set of elements.
    # State: A boolean array of size 4096 indicating which numbers are in the set.
    # Move: Flip a bit (add/remove a number).
    # Cost: Number of triple sum collisions.
    # We want to minimize collisions (to 0) and maximize the number of elements (or keep it high).
    #
    # This is a significant structural change from "greedy + incremental add/swap".
    #
    # Implementation details for SA:
    # 1. Initialize with the greedy set.
    # 2. Compute current collisions.
    # 3. Loop:
    #    a. Pick a random move (add/remove a random number).
    #    b. Calculate delta in collisions.
    #    c. Accept if delta < 0 or with probability exp(-delta/T).
    #    d. Cool down.
    #
    # Computing delta in collisions efficiently is key.
    # If we add x:
    #   - New sums involving x: 3x, 2x+s, x+s_i+s_j.
    #   - Check if these are in `used_sums`.
    #   - If yes, collision increases.
    #   - Also, we must check if the new sums collide with *each other*? No, sums involving x are distinct from each other if S is a set?
    #     3x vs 2x+s: 3x = 2x+s => x=s. Since x is new, x not in S, so distinct.
    #     2x+s1 vs 2x+s2: s1=s2. Distinct.
    #     2x+s vs x+s_i+s_j: 2x+s = x+s_i+s_j => x+s = s_i+s_j. Possible collision between a "2x+s" type and an "x+s_i+s_j" type.
    #     x+s_i+s_j vs x+s_k+s_l: s_i+s_j = s_k+s_l. Possible.
    #   So we need to check collisions between the new sums themselves.
    #
    # This is complex to compute delta for efficiently.
    #
    # Alternative: **Greedy with Lookahead / Backtracking**.
    # The current program has a greedy phase and a local search phase.
    # The local search in the current program is a bit slow due to Python loops.
    #
    # Let's optimize the local search using **NumPy** for the sum checks.
    #
    # Revised Plan:
    # 1. Greedy initialization (fast).
    # 2. Local Search using a "Conflict Count" array.
    #    - `used_sums` is a boolean array of size 12288.
    #    - `sum_count` is an int array of size 12288, counting how many times each sum appears.
    #    - A set is valid if all `sum_count[s] <= 1` for all s.
    #    - Collisions = sum(max(0, count[s]-1) for s).
    #
    #    When adding x:
    #    - Generate all new sums involving x.
    #    - For each new sum s:
    #        - If `sum_count[s] > 0`, it conflicts with existing.
    #        - Also check for conflicts among the new sums themselves.
    #
    #    This allows us to evaluate the "cost" of adding an element quickly.
    #
    # Let's implement a **Fast Local Search** that tries to add elements, and if stuck, removes one and tries again.
    #
    # Key optimization: Precompute all pairs (i, j) with i<=j.
    # For a set S, the sums are S[i]+S[j]+S[k].
    #
    # Let's try a different structural idea: **Genetic Algorithm**?
    # No, too slow in Python.
    #
    # Let's stick to **Optimized Local Search with NumPy**.
    #
    # We will represent the set as a sorted list.
    # We will maintain `sum_counts` as a numpy array.
    #
    # Function `add_element(S, x, sum_counts)`:
    #   - Generates new sums.
    #   - Checks for conflicts.
    #   - Returns (valid, new_sum_counts_delta) or just updates.
    #
    # Actually, the most effective improvement is likely **Parallel Greedy** or **Multiple Randomized Greedy** with a better stopping criterion.
    # The current program already does this.
    #
    # Let's try **Backtracking Search** with a strong pruning heuristic.
    # The inspiration 1 mentions backtracking.
    #
    # Let's combine:
    # 1. Greedy to get size N.
    # 2. Backtracking search to see if we can get N+1.
    #    - We fix the first k elements from the greedy solution.
    #    - We try to add more elements after that.
    #    - If we fail, we backtrack and try skipping one of the earlier elements?
    #
    # This is complex.
    #
    # Let's look at the "Bitset MIS" directive again.
    # If we interpret the problem as: "Find the largest subset of {1..4096} such that no two triples have the same sum".
    # This is a 3-wise constraint.
    #
    # However, there is a known result that for B_h sets, the problem is hard.
    #
    # Let's try a **Simulated Annealing** approach specifically designed for this.
    # It is structurally different from greedy/local search.
    #
    # State: `in_set` (boolean array of size 4096).
    # Objective: Maximize `sum(in_set)` subject to `collisions == 0`.
    #
    # To make it fast, we need to compute collisions quickly.
    # `collisions` can be computed by iterating over all triples? No, too slow.
    # We can compute `sum_counts` incrementally.
    #
    # Let's implement a **Fast Collision Counter**.
    #
    # Here is the plan for the new program:
    # 1. Greedy initialization.
    # 2. Simulated Annealing:
    #    - Start with the greedy set.
    #    - Define a move: pick a random index `i`. If `in_set[i]` is True, try to remove it. If False, try to add it.
    #    - Calculate the change in the number of valid elements and the number of collisions.
    #    - We accept a move if it reduces collisions, or if it maintains 0 collisions and increases size, or if it increases size but increases collisions (with some probability, to escape local minima).
    #    - Actually, we want to stay in the valid region (0 collisions) as much as possible.
    #    - So, primary goal: Keep collisions = 0. Secondary goal: Maximize size.
    #    - If a move keeps collisions = 0 and increases size, accept.
    #    - If a move keeps collisions = 0 and decreases size, reject (or accept with low prob).
    #    - If a move increases collisions, reject (or accept with low prob to escape).
    #    - If a move decreases collisions (from >0 to 0), accept.
    #
    #    We need a fast way to update `sum_counts` and `collisions` when adding/removing an element.
    #
    #    `add_element(x)`:
    #       - Generate list of new sums `new_sums`.
    #       - For each `s` in `new_sums`:
    #           - `old_count = sum_counts[s]`
    #           - `sum_counts[s] += 1`
    #           - If `old_count > 0`, collision increases by 1 (if old_count==0, no new collision, but if old_count>=1, the count goes from k to k+1, collisions increase by 1? No. Collisions = sum(count[s]-1). If count goes 0->1, delta=0. If 1->2, delta=1. If 2->3, delta=1.)
    #           - So, if `old_count >= 1`, collisions += 1.
    #       - Also, check for collisions *among* `new_sums`.
    #           - We need to count how many pairs of new sums are equal.
    #           - If `new_sums` has duplicates, that adds to collisions.
    #           - Wait, the `sum_counts` update above handles collisions with *existing* sums.
    #           - We also need to handle collisions *between* the new sums themselves.
    #           - If `new_sums` contains `s` twice, then when we process the first `s`, count goes 0->1 (no collision). When we process the second `s`, count goes 1->2 (collision +1).
    #           - So the simple loop `for s in new_sums: ...` actually handles both internal and external collisions correctly!
    #
    #    `remove_element(x)`:
    #       - Generate list of sums `old_sums` (same as new_sums for add).
    #       - For each `s` in `old_sums`:
    #           - `old_count = sum_counts[s]`
    #           - `sum_counts[s] -= 1`
    #           - If `old_count >= 2`, collisions -= 1. (Because count goes k->k-1. Collisions change from k-1 to k-2. Delta -1.)
    #           - If `old_count == 1`, collisions change from 0 to 0. Delta 0.
    #
    #    This is O(N^2) per move, which is fast enough for SA with ~1000-5000 moves.
    #
    #    We will run SA for a fixed number of steps or until time limit.
    #
    #    To maximize size, we can bias the moves:
    #    - If we are stuck (no improvement in size for X steps), try to remove a random element to "loosen" the constraints, then try to add a different one.
    #
    #    Let's implement this.

    def compute_initial_state(S):
        sum_counts = np.zeros(3 * max_val + 1, dtype=np.int32)
        for i in range(len(S)):
            for j in range(i, len(S)):
                for k in range(j, len(S)):
                    sum_counts[S[i] + S[j] + S[k]] += 1
        collisions = np.sum(np.maximum(sum_counts - 1, 0))
        return sum_counts, collisions

    def get_new_sums(S, x):
        """Yield all triple sums involving x and elements in S."""
        # 3x
        yield 3 * x
        # 2x + s
        for s in S:
            yield 2 * x + s
        # x + si + sj
        k = len(S)
        for i in range(k):
            for j in range(i, k):
                yield x + S[i] + S[j]

    # Initial State
    S = list(best_set)
    in_set = np.zeros(max_val + 1, dtype=bool)
    for s in S:
        in_set[s] = True
    
    sum_counts, collisions = compute_initial_state(S)
    current_size = len(S)
    best_size = current_size
    best_set_sa = list(S)

    # Simulated Annealing Parameters
    T = 1.0
    T_min = 0.01
    alpha = 0.999
    steps = 0
    max_steps = 50000 # Adjust based on time

    # Helper to update state
    def try_add(x, S, in_set, sum_counts, collisions):
        if in_set[x]:
            return False, collisions
        new_sums = list(get_new_sums(S, x))
        delta_coll = 0
        for s in new_sums:
            if sum_counts[s] >= 1:
                delta_coll += 1
            sum_counts[s] += 1
        # Accept?
        return True, collisions + delta_coll

    def try_remove(x, S, in_set, sum_counts, collisions):
        if not in_set[x]:
            return False, collisions
        new_sums = list(get_new_sums(S, x))
        delta_coll = 0
        for s in new_sums:
            if sum_counts[s] >= 2:
                delta_coll -= 1
            sum_counts[s] -= 1
        return True, collisions + delta_coll

    # Main SA Loop
    # We want to maximize size while keeping collisions 0.
    # We will allow some collisions temporarily to escape local minima.
    
    for step in range(max_steps):
        if time.time() - start_time > DEADLINE:
            break
        
        # Choose a move
        # 70% chance to try to add, 30% to try to remove (to balance)
        if random.random() < 0.7:
            # Try to add a random element not in S
            # Optimization: Pick from a list of candidates?
            # Just pick random x in 1..4096
            x = random.randint(1, max_val)
            if in_set[x]:
                continue
            
            # Calculate delta
            new_sums = list(get_new_sums(S, x))
            delta_coll = 0
            for s in new_sums:
                if sum_counts[s] >= 1:
                    delta_coll += 1
            
            if delta_coll == 0:
                # Valid move, increases size
                # Apply
                for s in new_sums:
                    sum_counts[s] += 1
                S.append(x)
                in_set[x] = True
                current_size += 1
                collisions += delta_coll
                if current_size > best_size:
                    best_size = current_size
                    best_set_sa = list(S)
            else:
                # Invalid move, increases collisions
                # Accept with probability exp(-delta_coll / T)
                if random.random() < np.exp(-delta_coll / T):
                    for s in new_sums:
                        sum_counts[s] += 1
                    S.append(x)
                    in_set[x] = True
                    current_size += 1
                    collisions += delta_coll
        else:
            # Try to remove a random element in S
            if current_size == 0:
                continue
            idx = random.randint(0, current_size - 1)
            x = S[idx]
            
            new_sums = list(get_new_sums(S, x))
            delta_coll = 0
            for s in new_sums:
                if sum_counts[s] >= 2:
                    delta_coll -= 1
            
            # Removing always decreases size.
            # Accept if it reduces collisions significantly or to escape.
            # If collisions > 0, we want to remove elements to fix collisions.
            # If collisions == 0, we rarely want to remove.
            
            if collisions > 0:
                # Accept if it reduces collisions
                if delta_coll < 0:
                    # Apply
                    for s in new_sums:
                        sum_counts[s] -= 1
                    S.pop(idx)
                    in_set[x] = False
                    current_size -= 1
                    collisions += delta_coll
                else:
                    # Even if it doesn't reduce collisions, maybe accept to shuffle?
                    if random.random() < np.exp(-(-delta_coll) / T): # exp(delta_coll/T)
                        for s in new_sums:
                            sum_counts[s] -= 1
                        S.pop(idx)
                        in_set[x] = False
                        current_size -= 1
                        collisions += delta_coll
            else:
                # Collisions are 0. Removing is bad.
                # Only accept with very low probability to explore?
                # Or if we are stuck?
                # Let's just reject to keep size high.
                pass

        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min

    # After SA, we might have a set with collisions > 0.
    # We need to clean it up to get a valid set.
    # Greedy cleanup: Remove elements that cause collisions.
    S_clean = list(best_set_sa)
    # Rebuild sum_counts for S_clean
    sum_counts_clean, collisions_clean = compute_initial_state(S_clean)
    
    # If collisions > 0, remove elements until valid
    # This is a simple heuristic: remove the element involved in the most collisions?
    # Or just remove random elements until valid.
    # Given SA tries to keep collisions low, this should be quick.
    
    while collisions_clean > 0 and time.time() - start_time < DEADLINE:
        # Find an element to remove
        # Just pick a random element
        if len(S_clean) == 0:
            break
        idx = random.randint(0, len(S_clean) - 1)
        x = S_clean[idx]
        
        # Calculate delta
        new_sums = list(get_new_sums(S_clean, x))
        delta_coll = 0
        for s in new_sums:
            if sum_counts_clean[s] >= 2:
                delta_coll -= 1
        
        # Apply removal
        for s in new_sums:
            sum_counts_clean[s] -= 1
        S_clean.pop(idx)
        collisions_clean += delta_coll

    # Final check
    if len(S_clean) > best_len:
        best_set = S_clean
        best_len = len(S_clean)
        
    # One more greedy pass on the final set to see if we can add more
    # (Similar to Phase 5 in original)
    S_final = list(best_set)
    if time.time() - start_time < DEADLINE - 5:
        used_sums = set()
        for i in range(len(S_final)):
            for j in range(i, len(S_final)):
                for k in range(j, len(S_final)):
                    used_sums.add(S_final[i] + S_final[j] + S_final[k])
        
        for x in range(1, max_val + 1):
            if time.time() - start_time > DEADLINE - 2:
                break
            if x in S_final:
                continue
            # Check can_add
            k = len(S_final)
            valid = True
            if (3*x) in used_sums:
                valid = False
            if valid:
                for s in S_final:
                    if (2*x + s) in used_sums:
                        valid = False
                        break
            if valid:
                for i in range(k):
                    for j in range(i, k):
                        if (x + S_final[i] + S_final[j]) in used_sums:
                            valid = False
                            break
                    if not valid:
                        break
            if valid:
                S_final.append(x)
                used_sums.add(3*x)
                for s in S_final[:-1]:
                    used_sums.add(2*x + s)
                for i in range(k):
                    for j in range(i, k):
                        used_sums.add(x + S_final[i] + S_final[j])
        
        if len(S_final) > best_len:
            best_set = S_final
            best_len = len(S_final)

    return best_set
