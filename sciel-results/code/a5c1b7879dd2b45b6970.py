import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    N = 180
    universe = list(range(1, N + 1))
    
    # 1. Precompute all forbidden triples (a, b, c) with a, b, c distinct
    # Condition: 1/a = 1/b + 1/c  =>  bc = a(b+c)
    # We can iterate a, b and solve for c:
    # bc = ab + ac => c(b-a) = ab => c = ab / (b-a)
    # So for each pair (a, b) with a != b, check if c is an integer in [1, N] and distinct from a, b.
    
    triples = []
    triple_set = set()
    
    for a in range(1, N + 1):
        for b in range(1, N + 1):
            if a == b:
                continue
            # c = ab / (b-a)
            # To avoid floating point issues, use integer arithmetic:
            # ab = c(b-a) => ab % (b-a) == 0
            denom = b - a
            if denom == 0:
                continue
            if (a * b) % denom == 0:
                c = (a * b) // denom
                if 1 <= c <= N and c != a and c != b:
                    # Store as a frozenset to avoid duplicates from permutations
                    t = frozenset((a, b, c))
                    if t not in triple_set:
                        triple_set.add(t)
                        triples.append(list(t))
    
    # 2. Build adjacency structures for fast conflict checking
    # For each element x, store the list of (y, z) such that {x, y, z} is a triple.
    # Also, for a pair (x, y), store the list of z such that {x, y, z} is a triple.
    
    neighbors = {i: [] for i in range(1, N + 1)}
    pair_to_third = {}
    
    for t in triples:
        a, b, c = t
        neighbors[a].append((b, c))
        neighbors[b].append((a, c))
        neighbors[c].append((a, b))
        
        pair_to_third[frozenset((a, b))] = c
        pair_to_third[frozenset((a, c))] = b
        pair_to_third[frozenset((b, c))] = a
        
    # 3. Simulated Annealing
    # State: A set S of elements.
    # Energy: Number of "active" triples (triples where all 3 elements are in S).
    # We want to minimize Energy to 0.
    
    # To make updates fast, we maintain:
    # - S: set of active elements
    # - conflict_count[x]: number of active triples involving x.
    # - total_conflicts: sum of conflict_count[x] / 3 (since each triple involves 3 elements).
    # Actually, it's easier to just count the number of active triples directly.
    # Let's maintain a set `active_triples`? No, that's slow to update.
    
    # Better approach:
    # Maintain `in_set` boolean array.
    # Maintain `conflict_degree[x]`: number of triples {x, y, z} such that y, z are also in S.
    # Total conflicts = sum(conflict_degree[x]) / 3.
    
    def initial_state():
        # Random sparse set or greedy?
        # Let's start with a random set of size ~ N/2 or use a greedy heuristic.
        # A good initial state is important.
        # Let's try a random permutation greedy fill first to get a valid set, then perturb.
        # Or just start random and let SA fix it? SA from random might be slow to reach 0 conflicts.
        # Let's start with a valid set from a quick greedy, then use SA to expand/improve.
        # Wait, if we start valid, we can only add elements if they don't create conflicts.
        # If we use 2-opt, we can remove 2 and add 2, potentially increasing size.
        
        # Let's generate a valid set using greedy with random order.
        S = set()
        order = random.sample(universe, N)
        in_set = [False] * (N + 1)
        conflict_degree = [0] * (N + 1)
        
        for x in order:
            # Check if adding x creates any conflict
            # A conflict is created if there exists (y, z) in neighbors[x] such that y, z in S.
            ok = True
            for y, z in neighbors[x]:
                if in_set[y] and in_set[z]:
                    ok = False
                    break
            if ok:
                S.add(x)
                in_set[x] = True
                # Update conflict degrees
                # For each (y, z) in neighbors[x], if y, z in S, then conflict_degree[x], [y], [z] increase?
                # Actually, we only add x if no such pair exists. So conflict_degree[x] remains 0.
                # But wait, if we add x, it might complete a triple with existing pairs?
                # No, we checked all (y,z) pairs involving x. If any (y,z) were in S, we wouldn't add x.
                # So adding a valid x keeps all conflict degrees 0?
                # Yes, if S was valid (0 conflicts), and we add x only if it doesn't form a triple with any pair in S,
                # then the new S is also valid.
        
        return S, in_set, conflict_degree

    def get_conflicts(S, in_set):
        """Calculate total number of active triples."""
        count = 0
        for t in triples:
            if t[0] in S and t[1] in S and t[2] in S:
                count += 1
        return count

    # For SA, we need fast evaluation of energy change for a move.
    # Move: Remove R={r1, r2}, Add A={a1, a2}.
    # We need to check if the new set is valid or has fewer conflicts.
    # Instead of maintaining complex degree counts which are hard to update for 2-opt,
    # we can just check the validity of the new set locally?
    # No, we need the total count.
    
    # Alternative: Use a bitmask or bitset for S? N=180 fits in 3 64-bit ints or Python int.
    # Python int is arbitrary precision, so we can represent S as a single integer bitset.
    # This makes "is x in S" very fast (bit check).
    # "Are a,b,c in S" is also fast.
    
    # Let's represent S as a Python integer.
    # bit i is set if element i is in S.
    
    def set_to_int(S):
        mask = 0
        for x in S:
            mask |= (1 << x)
        return mask

    def int_to_set(mask):
        return {i for i in range(1, N+1) if (mask >> i) & 1}

    def count_conflicts(mask):
        count = 0
        for t in triples:
            a, b, c = t
            if ((mask >> a) & 1) and ((mask >> b) & 1) and ((mask >> c) & 1):
                count += 1
        return count

    def is_valid(mask):
        return count_conflicts(mask) == 0

    best_mask = 0
    best_size = 0
    
    # Helper to update best
    def update_best(mask):
        nonlocal best_mask, best_size
        size = mask.bit_count()
        if size > best_size:
            best_size = size
            best_mask = mask

    # Phase 1: Multiple Greedy starts to find a good initial valid set
    # Greedy: add x if it doesn't create a conflict.
    # We can speed this up by precomputing for each x, the list of pairs.
    # But checking "does adding x create a conflict" requires checking all pairs (y,z) in neighbors[x]
    # and seeing if both y,z are in S.
    
    def greedy_random():
        mask = 0
        order = random.sample(universe, N)
        for x in order:
            # Check if adding x is safe
            safe = True
            for y, z in neighbors[x]:
                if ((mask >> y) & 1) and ((mask >> z) & 1):
                    safe = False
                    break
            if safe:
                mask |= (1 << x)
        return mask

    # Run a few greedy to find a large valid set
    for _ in range(20):
        if time.time() > deadline - 10:
            break
        m = greedy_random()
        if is_valid(m):
            update_best(m)

    # Phase 2: Simulated Annealing
    # Start from the best valid set found.
    # If no valid set found (unlikely), start from random.
    if best_mask == 0:
        best_mask = 1 << 1 # At least 1
    
    current_mask = best_mask
    current_energy = count_conflicts(current_mask) # Should be 0
    
    T = 10.0
    cooling = 0.999
    
    # Precompute list of elements for random selection
    elements = list(range(1, N + 1))
    
    while time.time() < deadline - 2:
        # Generate a 2-opt move
        # Remove 2, Add 2
        # Ensure we don't remove what we're adding or add what we're removing
        # Pick 4 distinct elements: r1, r2, a1, a2
        # r1, r2 must be in current_mask
        # a1, a2 must NOT be in current_mask
        
        # To ensure progress, we can bias towards adding elements that are "almost" compatible.
        # But random is fine for now.
        
        # Get list of elements in mask and not in mask
        # This is O(N) each time, which is fine for N=180.
        in_elements = [i for i in elements if (current_mask >> i) & 1]
        out_elements = [i for i in elements if not ((current_mask >> i) & 1)]
        
        if len(in_elements) < 2 or len(out_elements) < 2:
            break
        
        r1 = random.choice(in_elements)
        r2 = random.choice(in_elements)
        if r1 == r2:
            continue
            
        a1 = random.choice(out_elements)
        a2 = random.choice(out_elements)
        if a1 == a2:
            continue
        
        # Calculate new mask
        # Remove r1, r2
        # Add a1, a2
        new_mask = current_mask
        new_mask &= ~(1 << r1)
        new_mask &= ~(1 << r2)
        new_mask |= (1 << a1)
        new_mask |= (1 << a2)
        
        # Calculate energy change
        # old_energy = current_energy
        # new_energy = count_conflicts(new_mask)
        # This is O(E) where E is number of triples.
        # Number of triples is small. Let's estimate.
        # a,b up to 180. c = ab/(b-a).
        # Most pairs don't yield integer c.
        # Let's just count them.
        
        # Optimization: Only check triples involving r1, r2, a1, a2.
        # A triple is active if all 3 are in mask.
        # The set of active triples changes only if a triple involves at least one of r1, r2, a1, a2.
        # Let affected_triples = set of triples containing any of {r1, r2, a1, a2}.
        # We can precompute for each element x, the list of triples it belongs to.
        
        # Let's build that precomputation now.
        if not hasattr(construct, 'triple_adj'):
            construct.triple_adj = {i: [] for i in range(1, N + 1)}
            for idx, t in enumerate(triples):
                a, b, c = t
                construct.triple_adj[a].append(idx)
                construct.triple_adj[b].append(idx)
                construct.triple_adj[c].append(idx)
        
        affected_indices = set()
        for x in [r1, r2, a1, a2]:
            affected_indices.update(construct.triple_adj[x])
        
        # Calculate energy delta
        # For each affected triple, check if it was active before and if it is active now.
        delta = 0
        for idx in affected_indices:
            t = triples[idx]
            a, b, c = t
            was_active = ((current_mask >> a) & 1) and ((current_mask >> b) & 1) and ((current_mask >> c) & 1)
            is_active = ((new_mask >> a) & 1) and ((new_mask >> b) & 1) and ((new_mask >> c) & 1)
            if was_active and not is_active:
                delta -= 1
            elif not was_active and is_active:
                delta += 1
            # else no change
        
        new_energy = current_energy + delta
        
        # Acceptance probability
        if new_energy < current_energy:
            current_mask = new_mask
            current_energy = new_energy
            update_best(new_mask)
        else:
            # Metropolis criterion
            # P = exp(-delta / T)
            # delta > 0
            if T > 0:
                p = 2.718281828 ** (-delta / T) # Using e
                if random.random() < p:
                    current_mask = new_mask
                    current_energy = new_energy
                    # We don't update best if energy increased, but we might reach a better size later?
                    # Actually, if energy > 0, it's not a valid solution.
                    # We only care about valid solutions for the final answer.
                    # However, SA explores invalid states to escape local minima.
                    # We should track the best VALID solution found during the process.
                    # So we should check if new_energy == 0 and update best.
                    if new_energy == 0:
                        update_best(new_mask)
        
        T *= cooling
        if T < 0.01:
            # Reheat or restart?
            # Let's restart with a new random valid set or the best found so far.
            # Reheat
            T = 1.0
            
    return int_to_set(best_mask)

# Run construct to verify it works (optional, but good for debugging if run as script)
if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    # Verify
    valid = True
    res_set = set(res)
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            for k in range(j+1, len(res)):
                a, b, c = res[i], res[j], res[k]
                if 1/a == 1/b + 1/c:
                    valid = False
                    print(f"Conflict: {a}, {b}, {c}")
    print(f"Valid: {valid}")

