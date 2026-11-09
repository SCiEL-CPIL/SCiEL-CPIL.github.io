import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    # Target: 12 distinct integers <= 10000
    n = 12
    max_val = 10000
    
    # We want to minimize the number of distinct prime factors.
    # Let's try to find a solution with primes {2, 3, 5, 7}.
    # If that fails, we might try {2, 3, 5, 7, 11}, etc.
    # But {2,3,5,7} is a good first bet.
    
    primes_to_test = [
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7, 11, 13],
        [2, 3, 5, 7, 11, 13, 17]
    ]
    
    # Pre-compute smooth numbers for the first set of primes
    # A number is P-smooth if all its prime factors are in P.
    
    def is_smooth(x, primes_set):
        for p in primes_set:
            while x % p == 0:
                x //= p
        return x == 1
    
    # We will build the graph for the first prime set.
    # If we find a clique, we return it.
    # If not, we try the next set.
    
    for primes in primes_to_test:
        primes_set = set(primes)
        
        # Generate all P-smooth numbers <= 20000 (since max sum is 19998)
        smooth_numbers = set()
        # Recursively generate smooth numbers
        def gen_smooth(current, idx):
            if current > 20000:
                return
            smooth_numbers.add(current)
            if idx < len(primes):
                p = primes[idx]
                # Multiply by p repeatedly
                next_val = current
                while next_val <= 20000:
                    next_val *= p
                    gen_smooth(next_val, idx)
                # Move to next prime
                gen_smooth(current, idx + 1)
        
        gen_smooth(1, 0)
        
        # Build adjacency list using bitsets
        # Vertices are 1..10000
        # adj[i] is a bitmask of j such that i+j is smooth
        # Using Python integers as bitsets
        
        # To save memory and time, we only store bits for j > i? 
        # No, for clique search, it's easier to have full adjacency.
        # 10000 bits is about 1250 bytes per vertex. 10000 * 1250 = 12.5 MB. Acceptable.
        
        adj = [0] * (max_val + 1)
        
        # For each pair (i, j), check if i+j is smooth.
        # This is O(N^2) = 10^8 operations, which might be too slow in Python.
        # Optimization: Iterate over smooth sums S.
        # For each smooth S, for each i < S/2, j = S-i.
        # If i, j <= 10000, add edge.
        
        # Let's do this efficiently.
        for S in smooth_numbers:
            if S > 20000:
                continue
            # i + j = S
            # i < j to avoid duplicates, or just set both bits
            min_i = 1
            max_i = S - 1
            if max_i > max_val:
                max_i = max_val
            if min_i < 1:
                min_i = 1
            
            # Iterate i
            # We need to set bit j in adj[i] and bit i in adj[j]
            # j = S - i
            # j must be <= 10000 and j != i
            # So S - i <= 10000 => i >= S - 10000
            # And S - i >= 1 => i <= S - 1
            
            start_i = max(1, S - max_val)
            end_i = min(S - 1, max_val)
            
            for i in range(start_i, end_i + 1):
                j = S - i
                if i == j:
                    continue
                # Set bit j in adj[i]
                adj[i] |= (1 << j)
                # Set bit i in adj[j]
                adj[j] |= (1 << i)
                
        # Now find a clique of size n=12 in this graph.
        # We use a simple backtracking with bitsets.
        # Sort vertices by degree descending to prune earlier?
        # Or just try all starting vertices.
        
        found_clique = None
        
        def find_clique(candidates, current_clique, depth):
            if time.time() > deadline:
                return None
            if len(current_clique) == n:
                return current_clique[:]
            
            # If not enough candidates left
            if len(candidates) < n - len(current_clique):
                return None
            
            # Pick a vertex from candidates
            # Heuristic: pick one with high degree within candidates?
            # For simplicity, pick the first one, or random.
            # Let's pick the one with the smallest index to be deterministic?
            # Or the one with the most neighbors in candidates.
            
            # To speed up, we can just iterate.
            # We need to choose a vertex v from candidates.
            # New candidates = candidates & adj[v] & (~current_clique_mask)
            
            # Let's iterate through candidates
            # We can pick the candidate with the highest "score" (degree in candidates)
            # But calculating that is costly. Let's just pick the first available.
            # Actually, branching factor is crucial.
            
            # Let's try to pick a vertex that is connected to the most other candidates.
            # This is a heuristic for MaxClique.
            
            # For simplicity and speed, let's just pick the first one.
            # If the graph is dense, this works.
            
            # Get the lowest set bit in candidates
            if candidates == 0:
                return None
            
            # Pick the lowest set bit
            # v = (candidates & -candidates).bit_length() - 1
            # This is O(1) for bit ops.
            
            # However, picking the lowest index might not be optimal for search depth.
            # Let's pick a random one? No, deterministic is better for reproducibility.
            # Let's pick the one with the highest degree in the candidate set.
            # This requires iterating all candidates.
            
            # Let's stick to the first one for now, but maybe sort candidates initially?
            # The input order is 1..10000.
            
            # Let's try picking the vertex with the maximum number of neighbors in `candidates`.
            # This is a standard heuristic.
            
            # But iterating all candidates to find max degree is O(C).
            # If C is large, this is slow.
            
            # Alternative: Just pick the first one.
            # If it fails, backtrack.
            
            # Let's optimize the selection:
            # We'll iterate through the set bits of candidates.
            # We'll keep track of the best candidate.
            
            best_v = -1
            best_deg = -1
            temp_cands = candidates
            while temp_cands:
                lsb = temp_cands & -temp_cands
                v = lsb.bit_length() - 1
                # Count bits in (candidates & adj[v])
                deg = (candidates & adj[v]).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                temp_cands ^= lsb
                # Early termination if we find a perfect connection?
                if best_deg == candidates.bit_count() - 1:
                    break
            
            if best_v == -1:
                return None
                
            v = best_v
            current_clique.append(v)
            
            # New candidates: intersection of current candidates and neighbors of v
            # Exclude v itself? adj[v] doesn't have self-loop, so it's fine.
            new_candidates = candidates & adj[v]
            
            # Recursive call
            result = find_clique(new_candidates, current_clique, depth + 1)
            if result:
                return result
            
            current_clique.pop()
            
            # Remove v from candidates for next iteration at this level
            # We passed `candidates` by value (int is immutable), so we just return.
            # The caller will loop? No, this function picks ONE vertex.
            # We need to loop over choices of v.
            
            # The current structure only tries ONE path (greedy choice).
            # We need backtracking over the choice of v.
            
            # Let's restructure find_clique to loop over all possible v in candidates.
            
            return None
        
        # Revised Find Clique with backtracking over vertex choice
        def find_clique_backtrack(candidates, current_clique):
            if time.time() > deadline:
                return None
            if len(current_clique) == n:
                return current_clique[:]
            
            if candidates == 0:
                return None
            
            # If not enough candidates
            if candidates.bit_count() < n - len(current_clique):
                return None
            
            # Iterate through candidates
            # We'll pick vertices in some order.
            # To improve success rate, we can sort by degree in candidates?
            # But that's expensive. Let's just iterate.
            
            temp_cands = candidates
            while temp_cands:
                if time.time() > deadline:
                    return None
                lsb = temp_cands & -temp_cands
                v = lsb.bit_length() - 1
                temp_cands ^= lsb
                
                current_clique.append(v)
                new_candidates = candidates & adj[v]
                
                # We must only consider candidates that are AFTER v? 
                # No, the graph is undirected. 
                # But to avoid duplicates, we usually enforce an order.
                # If we enforce v1 < v2 < ... < vk, then new_candidates should only contain vertices > v.
                # This reduces the search space significantly.
                
                # Let's enforce ordering: all vertices in clique must be increasing.
                # So new_candidates = candidates & adj[v] & (mask of vertices > v)
                
                # Mask of vertices > v:
                # All bits set from v+1 to max_val
                # mask = ((1 << (max_val + 1)) - 1) ^ ((1 << (v + 1)) - 1)
                # This is expensive to compute each time.
                # Pre-compute suffix masks?
                # suffix_mask[v] = bits set for all j > v
                
                # Alternatively, just intersect with candidates (which already excludes previous ones? No).
                # In the standard Bron-Kerbosch with pivot, we maintain the candidate set.
                # If we pick v, the next candidates must be neighbors of v that are in the current candidate set.
                # If we don't enforce order, we might find the same clique multiple times.
                # But we just need ONE clique.
                # If we don't enforce order, we might cycle? No, we remove v from candidates in the next level?
                # No, `new_candidates` is a subset of `candidates`.
                # If we don't remove v, we might pick v again? No, adj[v] doesn't have v.
                # But we might pick a neighbor w, then w's neighbor v?
                # Yes, that's a different path to the same set, or a cycle.
                # To prevent this, we should enforce an ordering.
                
                # Let's enforce ordering: next vertices must be > v.
                # This ensures each clique is found exactly once (in sorted order).
                
                # Efficiently get mask of bits > v
                # We can pre-compute this.
                
                result = find_clique_backtrack(new_candidates & suffix_mask[v], current_clique)
                if result:
                    return result
                
                current_clique.pop()
                
            return None
        
        # Pre-compute suffix masks
        # suffix_mask[v] has bits set for all j in [v+1, max_val]
        suffix_mask = [0] * (max_val + 2)
        # suffix_mask[max_val] = 0
        # suffix_mask[v] = suffix_mask[v+1] | (1 << (v+1)) ... wait
        # We want bits for j > v.
        # Let's build from high to low.
        # mask = 0
        # for v from max_val down to 1:
        #    suffix_mask[v] = mask
        #    mask |= (1 << v)  # No, we want j > v.
        # 
        # Let's do:
        # all_bits = (1 << (max_val + 1)) - 1  # bits 0..max_val
        # mask_high = all_bits
        # for v in range(max_val, 0, -1):
        #    # suffix_mask[v] should have bits for v+1 ... max_val
        #    # Remove bit v from the "available higher" set?
        #    # Actually, simpler:
        #    suffix_mask[v] = all_bits ^ ((1 << (v + 1)) - 1)
        # This is O(1) per v if we just do the math.
        # But (1 << (v+1)) is a big int.
        # Let's precompute powers of 2? No, just use the formula.
        
        # Optimization:
        # We can just pass `candidates` and ensure we only pick from it.
        # If we enforce v1 < v2 < ..., then when we pick v, the next candidates must be > v.
        # So `new_candidates = candidates & adj[v] & suffix_mask[v]`
        
        # Let's initialize the search
        # Initial candidates: all vertices 1..max_val
        # We can restrict to vertices that have degree >= n-1
        # This prunes the initial set.
        
        initial_candidates = 0
        for i in range(1, max_val + 1):
            if adj[i].bit_count() >= n - 1:
                initial_candidates |= (1 << i)
        
        # Start search
        clique = find_clique_backtrack(initial_candidates, [])
        
        if clique:
            # Verify the clique
            # Check all pairs
            valid = True
            for i in range(len(clique)):
                for j in range(i+1, len(clique)):
                    s = clique[i] + clique[j]
                    if not is_smooth(s, primes_set):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                return sorted(clique)
            else:
                # Should not happen if logic is correct
                pass
                
        # If no clique found for this prime set, try next one
        # Note: The graph construction is expensive. 
        # If the first set fails, it might be because no 12-clique exists.
        # Or the search timed out.
        # If it timed out, we should probably return the best found so far?
        # But we didn't find any.
        # We can return a heuristic solution if we run out of time.
        
    # Fallback: Return a simple set, e.g., 1..12
    # This ensures we don't crash.
    return list(range(1, 13))

# Pre-compute suffix masks globally to avoid re-computation in multiple calls?
# construct() is called once per evaluation.
# But we can define them inside or globally.
# Let's define them inside the loop where they are used, or globally.
# Since max_val is fixed, we can compute them once globally.

MAX_VAL = 10000
ALL_BITS = (1 << (MAX_VAL + 1)) - 1
SUFFIX_MASK = [0] * (MAX_VAL + 2)
for v in range(MAX_VAL, 0, -1):
    # Bits set for indices > v
    # Indices are 1..MAX_VAL
    # We want bits v+1 .. MAX_VAL
    # (1 << (v+1)) - 1 gives bits 0..v
    # ALL_BITS ^ ((1 << (v+1)) - 1) gives bits v+1 .. MAX_VAL
    SUFFIX_MASK[v] = ALL_BITS ^ ((1 << (v + 1)) - 1)
SUFFIX_MASK[MAX_VAL] = 0

# Re-define construct to use global SUFFIX_MASK
def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    n = 12
    max_val = 10000
    
    primes_to_test = [
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7, 11, 13]
    ]
    
    def is_smooth(x, primes_set):
        for p in primes_set:
            while x % p == 0:
                x //= p
        return x == 1

    for primes in primes_to_test:
        primes_set = set(primes)
        
        smooth_numbers = set()
        def gen_smooth(current, idx):
            if current > 20000:
                return
            smooth_numbers.add(current)
            if idx < len(primes):
                p = primes[idx]
                next_val = current
                while next_val <= 20000:
                    next_val *= p
                    gen_smooth(next_val, idx)
                gen_smooth(current, idx + 1)
        
        gen_smooth(1, 0)
        
        adj = [0] * (max_val + 1)
        
        for S in smooth_numbers:
            if S > 20000:
                continue
            start_i = max(1, S - max_val)
            end_i = min(S - 1, max_val)
            
            for i in range(start_i, end_i + 1):
                j = S - i
                if i == j:
                    continue
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        
        def find_clique_backtrack(candidates, current_clique):
            if time.time() > deadline:
                return None
            if len(current_clique) == n:
                return current_clique[:]
            
            if candidates == 0:
                return None
            
            if candidates.bit_count() < n - len(current_clique):
                return None
            
            temp_cands = candidates
            while temp_cands:
                if time.time() > deadline:
                    return None
                lsb = temp_cands & -temp_cands
                v = lsb.bit_length() - 1
                temp_cands ^= lsb
                
                current_clique.append(v)
                new_candidates = candidates & adj[v] & SUFFIX_MASK[v]
                
                result = find_clique_backtrack(new_candidates, current_clique)
                if result:
                    return result
                
                current_clique.pop()
                
            return None
        
        initial_candidates = 0
        for i in range(1, max_val + 1):
            if adj[i].bit_count() >= n - 1:
                initial_candidates |= (1 << i)
        
        clique = find_clique_backtrack(initial_candidates, [])
        
        if clique:
            valid = True
            for i in range(len(clique)):
                for j in range(i+1, len(clique)):
                    s = clique[i] + clique[j]
                    if not is_smooth(s, primes_set):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                return sorted(clique)

    return list(range(1, 13))
