import time
import random
import sys
from itertools import combinations

def construct():
    # Set a hard deadline with safety margin
    deadline = time.time() + 285.0
    
    # 1. Generate Paley Graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    n = q
    
    # Helper to get K4s and Alpha4s (K4 in complement)
    def get_k4s(adj, n):
        k4s = []
        for a in range(n):
            for b in range(a + 1, n):
                if not (adj[a] >> b) & 1:
                    continue
                for c in range(b + 1, n):
                    if not ((adj[a] >> c) & 1 and (adj[b] >> c) & 1):
                        continue
                    for d in range(c + 1, n):
                        if ((adj[a] >> d) & 1 and (adj[b] >> d) & 1 and (adj[c] >> d) & 1):
                            mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                            k4s.append(mask)
        return k4s

    def get_alpha4s(adj, n):
        alpha4s = []
        for a in range(n):
            for b in range(a + 1, n):
                if (adj[a] >> b) & 1: # If edge exists, not in independent set
                    continue
                for c in range(b + 1, n):
                    if ((adj[a] >> c) & 1 or (adj[b] >> c) & 1):
                        continue
                    for d in range(c + 1, n):
                        if not ((adj[a] >> d) & 1 or (adj[b] >> d) & 1 or (adj[c] >> d) & 1):
                            mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                            alpha4s.append(mask)
        return alpha4s

    # Initial lists
    k4_list = get_k4s(adj, n)
    alpha4_list = get_alpha4s(adj, n)
    
    # Helper to build maps for fast incremental updates
    def build_maps(k4_list, alpha4_list, n):
        k4_by_vertex = [[] for _ in range(n)]
        alpha4_by_vertex = [[] for _ in range(n)]
        for k4 in k4_list:
            m = k4
            while m:
                lsb = m & -m
                idx = (lsb.bit_length() - 1)
                k4_by_vertex[idx].append(k4)
                m ^= lsb
        for a4 in alpha4_list:
            m = a4
            while m:
                lsb = m & -m
                idx = (lsb.bit_length() - 1)
                alpha4_by_vertex[idx].append(a4)
                m ^= lsb
        return k4_by_vertex, alpha4_by_vertex

    k4_by_vertex, alpha4_by_vertex = build_maps(k4_list, alpha4_list, n)
    
    # Optimized search for a valid neighbor set for a new vertex
    # Uses incremental score updates based on precomputed maps
    def find_valid_N_opt(n, k4_list, alpha4_list, k4_by_vertex, alpha4_by_vertex, time_limit=5.0):
        start_time = time.time()
        best_score = float('inf')
        best_N = None
        full_mask = (1 << n) - 1
        
        # Try a few random restarts
        for restart in range(30):
            if time.time() > start_time + time_limit:
                break
                
            # Initialize N randomly
            N = 0
            for i in range(n):
                if random.random() < 0.5:
                    N |= (1 << i)
            
            # Calculate initial score
            current_score = 0
            for k4 in k4_list:
                if (N & k4) == k4:
                    current_score += 1
            not_N = full_mask & ~N
            for a4 in alpha4_list:
                if (not_N & a4) == a4:
                    current_score += 1
            
            if current_score == 0:
                return N
            
            if current_score < best_score:
                best_score = current_score
                best_N = N
            
            # Hill climbing with occasional random jumps
            T = 1.0
            while time.time() - start_time < time_limit:
                if current_score == 0:
                    return N
                
                # Pick a random bit to flip
                bit = random.randint(0, n - 1)
                delta_bit = 1 << bit
                new_N = N ^ delta_bit
                
                # Incremental update
                added = (new_N & delta_bit) != 0
                delta_score = 0
                
                if added:
                    # Check K4s containing bit that are now fully in N
                    for k4 in k4_by_vertex[bit]:
                        if (new_N & k4) == k4 and (N & k4) != k4:
                            delta_score += 1
                    # Check Alpha4s containing bit that are now fully in N (so broken in ~N)
                    for a4 in alpha4_by_vertex[bit]:
                        if (N & a4) == 0:
                            delta_score -= 1
                else:
                    # Removing bit
                    # K4s: If bit was in N and in k4, and all other bits of k4 were in N, then it was a violation. Now it's not.
                    for k4 in k4_by_vertex[bit]:
                        if (N & k4) == k4 and (new_N & k4) != k4:
                            delta_score -= 1
                    # Alpha4s: If bit was in ~N and in a4, and all other bits of a4 were in ~N, then it was a violation.
                    # Now bit is in N, so a4 has a bit in N, so it's not a violation?
                    # Wait. Alpha4 violation: a4 subset of ~N.
                    # If we remove bit from N (add to ~N), we might complete an Alpha4.
                    # Condition: a4 subset of ~N_new.
                    # ~N_new = ~N_old | delta_bit.
                    # So a4 subset of ~N_old | delta_bit.
                    # If a4 contains bit, and a4 \ {bit} is subset of ~N_old, then a4 is now a violation.
                    # (new_N & a4) == 0?
                    # new_N = N ^ delta_bit (bit removed).
                    # So (new_N & a4) == 0 means a4 has no bits in new_N.
                    # Since bit is not in new_N, this is equivalent to (N & (a4 ^ delta_bit)) == 0?
                    # Yes, if bit is in a4.
                    for a4 in alpha4_by_vertex[bit]:
                        if (new_N & a4) == 0 and (N & a4) != 0:
                            # It was not a violation before (N had a bit from a4, which must be bit since new_N doesn't have it? No, N had bit).
                            # If N had bit, and new_N doesn't, and new_N has no other bits from a4, then N had exactly bit from a4.
                            # So (N & a4) == delta_bit.
                            if (N & a4) == delta_bit:
                                delta_score += 1
                
                new_score = current_score + delta_score
                
                if new_score < 0: new_score = 0 # Safety
                
                if new_score < current_score or random.random() < pow(2.718, -(new_score - current_score) / T):
                    N = new_N
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_N = N
                T *= 0.995 # Cool down slowly
            
            if best_score == 0:
                return best_N

        return best_N if best_score == 0 else None

    # Extension loop
    for target_n in range(n + 1, 47):
        if time.time() > deadline - 15.0:
            break
        
        # Time budget per vertex
        time_per_vertex = 4.0
        if target_n > 40:
            time_per_vertex = 8.0
        
        found_N = find_valid_N_opt(n, k4_list, alpha4_list, k4_by_vertex, alpha4_by_vertex, time_per_vertex)
        
        if found_N is None:
            # Try a few more times with different random seeds or longer time?
            # If we fail, we might be stuck.
            break
        
        # Add the new vertex
        new_vertex = n
        adj.append(0)
        for i in range(n):
            if (found_N >> i) & 1:
                adj[new_vertex] |= (1 << i)
                adj[i] |= (1 << new_vertex)
        
        n += 1
        
        # Update k4_list and alpha4_list
        new_k4s = []
        neigh_mask = adj[new_vertex]
        # Iterate over triples in neigh_mask
        for a in range(n - 1):
            if not (neigh_mask >> a) & 1:
                continue
            for b in range(a + 1, n - 1):
                if not (neigh_mask >> b) & 1:
                    continue
                if not (adj[a] >> b) & 1:
                    continue
                for c in range(b + 1, n - 1):
                    if not (neigh_mask >> c) & 1:
                        continue
                    if ((adj[a] >> c) & 1) and ((adj[b] >> c) & 1):
                        mask = (1 << a) | (1 << b) | (1 << c) | (1 << new_vertex)
                        new_k4s.append(mask)
        
        k4_list.extend(new_k4s)
        
        new_alpha4s = []
        non_neigh_mask = ((1 << (n - 1)) - 1) & ~neigh_mask
        for a in range(n - 1):
            if not (non_neigh_mask >> a) & 1:
                continue
            for b in range(a + 1, n - 1):
                if not (non_neigh_mask >> b) & 1:
                    continue
                if (adj[a] >> b) & 1: # Edge exists, so not independent
                    continue
                for c in range(b + 1, n - 1):
                    if not (non_neigh_mask >> c) & 1:
                        continue
                    if ((adj[a] >> c) & 1) or ((adj[b] >> c) & 1):
                        continue
                    mask = (1 << a) | (1 << b) | (1 << c) | (1 << new_vertex)
                    new_alpha4s.append(mask)
        
        alpha4_list.extend(new_alpha4s)
        
        # Update maps
        # We can just rebuild for simplicity, as n is small and lists are not huge yet.
        # Or update incrementally.
        # Rebuild is O(n * total_constraints).
        k4_by_vertex, alpha4_by_vertex = build_maps(k4_list, alpha4_list, n)

    # Convert adj to edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                edges.append((i, j))
    
    return n, edges
