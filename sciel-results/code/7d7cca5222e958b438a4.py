"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    DEADLINE = 58.0  # Safety margin before 60s limit
    
    N = 256
    vertices = list(range(1, N + 1))
    
    # Precompute perfect squares up to 2*N
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1
    
    # Build conflict graph using bitmasks
    # adj[i] is a bitmask of vertices j such that (i+1) + (j+1) is a perfect square
    adj = [0] * N
    
    for i in range(N):
        val = i + 1
        for j in range(N):
            other = j + 1
            if val + other in squares:
                adj[i] |= (1 << j)
    
    # Identify vertices that cannot be in any valid set (self-conflict: 2*a is a square)
    self_conflict = [False] * N
    for i in range(N):
        if adj[i] & (1 << i):
            self_conflict[i] = True
    
    # Candidate vertices (no self-conflict)
    candidates = [i for i in range(N) if not self_conflict[i]]
    
    # Precompute neighbor masks restricted to candidates
    candidate_mask = 0
    for i in candidates:
        candidate_mask |= (1 << i)
    
    # neighbors[i] contains only valid candidates that conflict with i
    neighbors = [adj[i] & candidate_mask for i in range(N)]
    
    def greedy_independent_set(order):
        """Greedy: process vertices in given order, add if no conflict with already added."""
        selected_mask = 0
        for v in order:
            if selected_mask & neighbors[v]:
                continue
            selected_mask |= (1 << v)
        return selected_mask

    def degree_based_order(cand_list):
        """Order by degree (number of conflicts within candidates) ascending."""
        degrees = []
        for v in cand_list:
            deg = bin(neighbors[v]).count('1')
            degrees.append((deg, v))
        degrees.sort()
        return [v for _, v in degrees]

    def local_search_1_2_exchange(selected_mask, max_iters=1000):
        """
        Local search using 1-2 exchanges.
        Tries to replace one vertex u in S with two vertices v, w not in S,
        such that v, w are compatible with S\{u} and with each other.
        """
        best_mask = selected_mask
        best_size = bin(selected_mask).count('1')
        
        for _ in range(max_iters):
            if time.time() - start_time > DEADLINE:
                break
            
            improved = False
            selected_list = [i for i in range(N) if best_mask & (1 << i)]
            random.shuffle(selected_list)
            
            for u in selected_list:
                # Remove u from the set
                mask_without_u = best_mask & ~(1 << u)
                
                # Calculate neighbors of mask_without_u
                # We need to find vertices v, w that are NOT neighbors of any node in mask_without_u
                # and NOT in mask_without_u themselves.
                
                # Efficient way to compute N(mask_without_u):
                # It's the union of neighbors of all bits set in mask_without_u.
                # Since N is small (256), we can iterate.
                union_neighbors = 0
                temp_mask = mask_without_u
                while temp_mask:
                    # Get lowest set bit
                    lsb = temp_mask & -temp_mask
                    idx = (lsb.bit_length() - 1)
                    union_neighbors |= neighbors[idx]
                    temp_mask ^= lsb
                    
                # Available vertices are those in candidate_mask, not in mask_without_u, 
                # and not in union_neighbors.
                available = candidate_mask & ~mask_without_u & ~union_neighbors
                
                # We need to find two vertices v, w in 'available' such that they are not connected.
                # i.e., (neighbors[v] & (1<<w)) == 0 and (neighbors[w] & (1<<v)) == 0
                # Since graph is undirected, just check one.
                
                avail_list = [i for i in range(N) if available & (1 << i)]
                
                # Optimization: if len(avail_list) < 2, skip
                if len(avail_list) < 2:
                    continue
                
                # Check for independent pair in avail_list
                # Randomize to avoid bias
                random.shuffle(avail_list)
                
                found_pair = False
                for i in range(len(avail_list)):
                    v = avail_list[i]
                    # v is compatible with mask_without_u by construction
                    # We need w such that w is in avail_list, w != v, and v-w is not an edge
                    # Also w must be compatible with mask_without_u (guaranteed by avail_list)
                    
                    # We want to add v and w.
                    # So we need v and w to not conflict with each other.
                    # And they are already known to not conflict with mask_without_u.
                    
                    # To speed up, we can just pick the first w that doesn't conflict with v.
                    # But we want to ensure we find a pair.
                    for j in range(i + 1, len(avail_list)):
                        w = avail_list[j]
                        if not (neighbors[v] & (1 << w)):
                            # Found a valid pair (v, w)
                            new_mask = mask_without_u | (1 << v) | (1 << w)
                            new_size = bin(new_mask).count('1')
                            if new_size > best_size:
                                best_mask = new_mask
                                best_size = new_size
                                improved = True
                                found_pair = True
                                break
                    if found_pair:
                        break
                
            if not improved:
                # If no 1-2 exchange worked, try simple random restarts or break?
                # For now, break if no improvement in a full pass.
                # We could also try 1-1 exchange with a different vertex if we weren't maximal, 
                # but we maintain maximality.
                break
                
        return best_mask, best_size

    # Strategy 1: Degree-based greedy
    best_mask = 0
    best_size = 0
    
    # Try degree-based ordering
    order = degree_based_order(candidates)
    selected_mask = greedy_independent_set(order)
    best_size = bin(selected_mask).count('1')
    best_mask = selected_mask
    
    # Local search on the greedy result
    improved_mask, improved_size = local_search_1_2_exchange(best_mask, max_iters=200)
    if improved_size > best_size:
        best_size = improved_size
        best_mask = improved_mask

    # Strategy 2: Multiple random restarts
    # We use random orderings for the initial greedy set, then refine.
    while time.time() - start_time < DEADLINE:
        # Random ordering
        random_order = list(candidates)
        random.shuffle(random_order)
        
        selected_mask = greedy_independent_set(random_order)
        current_size = bin(selected_mask).count('1')
        
        # Only run local search if this greedy result is promising or better than current
        # Actually, always run it but limit iterations if time is tight
        if current_size >= best_size - 5: # Heuristic: don't waste time on very bad starts
             improved_mask, improved_size = local_search_1_2_exchange(selected_mask, max_iters=50)
             if improved_size > best_size:
                 best_size = improved_size
                 best_mask = improved_mask
                 
        # Check time
        if time.time() - start_time > DEADLINE:
            break

    # Convert back to values
    result = [i + 1 for i in range(N) if best_mask & (1 << i)]
    
    # Final validation (should be fast for N=256)
    # The problem guarantees we need a valid set.
    # If result is empty, return a single valid vertex if possible
    if not result:
        for i in candidates:
            return [i+1]
        return [1] # Fallback

    return result
# EVOLVE-BLOCK-END
