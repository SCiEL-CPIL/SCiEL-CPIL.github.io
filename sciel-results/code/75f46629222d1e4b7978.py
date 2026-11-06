import time
import random
import sys

def construct():
    # Set a hard deadline
    deadline = time.time() + 290.0
    
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
    
    # Precompute incidence lists for fast delta calculation
    # k4_incidence[v] = list of indices of k4s in k4_list that contain vertex v
    # alpha_incidence[v] = list of indices of alpha4s in alpha4_list that contain vertex v
    k4_incidence = [[] for _ in range(n)]
    for idx, mask in enumerate(k4_list):
        v = 0
        while mask:
            if mask & 1:
                k4_incidence[v].append(idx)
            v += 1
            mask >>= 1
            
    alpha_incidence = [[] for _ in range(n)]
    for idx, mask in enumerate(alpha4_list):
        v = 0
        while mask:
            if mask & 1:
                alpha_incidence[v].append(idx)
            v += 1
            mask >>= 1

    # Function to calculate score (number of violations) for a given N
    # N is a bitmask of neighbors in the current graph (0..n-1)
    def calc_score(N, n, k4_list, alpha4_list):
        score = 0
        full_mask = (1 << n) - 1
        not_N = full_mask & ~N
        for k4 in k4_list:
            if (N & k4) == k4:
                score += 1
        for a4 in alpha4_list:
            if (not_N & a4) == a4:
                score += 1
        return score

    # Local search to find a valid N
    def find_valid_N(n, k4_list, alpha4_list, k4_inc, alpha_inc, time_limit=5.0):
        start_time = time.time()
        best_score = float('inf')
        best_N = None
        full_mask = (1 << n) - 1
        
        # Try a few random restarts
        for restart in range(50):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize N randomly, roughly half bits set
            N = 0
            for i in range(n):
                if random.random() < 0.5:
                    N |= (1 << i)
            
            current_score = calc_score(N, n, k4_list, alpha4_list)
            
            if current_score == 0:
                return N
            
            if current_score < best_score:
                best_score = current_score
                best_N = N
            
            # Hill climbing / SA
            T = 1.0
            no_improve = 0
            while time.time() - start_time < time_limit:
                if current_score == 0:
                    return N
                
                # Pick a random bit to flip
                bit = random.randint(0, n - 1)
                delta_bit = 1 << bit
                
                new_N = N ^ delta_bit
                
                # Calculate delta in score efficiently
                # Only K4s/Alpha4s containing 'bit' can change status
                delta_score = 0
                not_N = full_mask & ~N
                new_not_N = full_mask & ~new_N
                
                # Check K4s
                for idx in k4_inc[bit]:
                    k4 = k4_list[idx]
                    # Was it a violation?
                    old_viol = 1 if (N & k4) == k4 else 0
                    # Is it a violation now?
                    new_viol = 1 if (new_N & k4) == k4 else 0
                    delta_score += new_viol - old_viol
                    
                # Check Alpha4s
                for idx in alpha_inc[bit]:
                    a4 = alpha4_list[idx]
                    old_viol = 1 if (not_N & a4) == a4 else 0
                    new_viol = 1 if (new_not_N & a4) == a4 else 0
                    delta_score += new_viol - old_viol
                
                new_score = current_score + delta_score
                
                if new_score < current_score:
                    N = new_N
                    current_score = new_score
                    no_improve = 0
                    if current_score < best_score:
                        best_score = current_score
                        best_N = N
                    T = max(0.1, T * 0.95) # Cool down on improvement
                else:
                    # Accept with probability exp(-delta/T)
                    if random.random() < pow(2.718, -delta_score / T):
                        N = new_N
                        current_score = new_score
                        no_improve += 1
                    else:
                        no_improve += 1
                    T = max(0.01, T * 0.99) # Cool down always
                    if no_improve > 100:
                        break # Stuck, restart

            if best_score == 0:
                return best_N

        return best_N if best_score == 0 else None

    # Extension loop
    for target_n in range(n + 1, 47):
        if time.time() > deadline - 10.0:
            break
        
        # Time budget per vertex
        time_per_vertex = 5.0
        if target_n > 40:
            time_per_vertex = 10.0
        
        found_N = find_valid_N(n, k4_list, alpha4_list, k4_incidence, alpha_incidence, time_per_vertex)
        
        if found_N is None:
            break
        
        # Add the new vertex
        new_vertex = n
        adj.append(0)
        for i in range(n):
            if (found_N >> i) & 1:
                adj[new_vertex] |= (1 << i)
                adj[i] |= (1 << new_vertex)
        
        n += 1
        
        # Update k4_list and alpha_list
        new_k4s = []
        neigh_mask = adj[new_vertex]
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
        
        # Extend incidence lists for new K4s
        for mask in new_k4s:
            k4_list.append(mask)
            idx = len(k4_list) - 1
            v = 0
            while mask:
                if mask & 1:
                    if v < len(k4_incidence):
                        k4_incidence[v].append(idx)
                    else:
                        k4_incidence.append([idx])
                v += 1
                mask >>= 1

        # Ensure k4_incidence has enough entries
        while len(k4_incidence) < n:
            k4_incidence.append([])
            
        # Update Alpha4s
        new_alpha4s = []
        non_neigh_mask = ((1 << (n - 1)) - 1) & ~neigh_mask
        for a in range(n - 1):
            if not (non_neigh_mask >> a) & 1:
                continue
            for b in range(a + 1, n - 1):
                if not (non_neigh_mask >> b) & 1:
                    continue
                if (adj[a] >> b) & 1:
                    continue
                for c in range(b + 1, n - 1):
                    if not (non_neigh_mask >> c) & 1:
                        continue
                    if ((adj[a] >> c) & 1) or ((adj[b] >> c) & 1):
                        continue
                    mask = (1 << a) | (1 << b) | (1 << c) | (1 << new_vertex)
                    new_alpha4s.append(mask)
                    
        for mask in new_alpha4s:
            alpha4_list.append(mask)
            idx = len(alpha4_list) - 1
            v = 0
            while mask:
                if mask & 1:
                    if v < len(alpha_incidence):
                        alpha_incidence[v].append(idx)
                    else:
                        alpha_incidence.append([idx])
                v += 1
                mask >>= 1

        while len(alpha_incidence) < n:
            alpha_incidence.append([])

    # Convert adj to edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                edges.append((i, j))
    
    return n, edges
