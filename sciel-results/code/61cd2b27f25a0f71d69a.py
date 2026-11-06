import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 295  # Safety margin before 330s kill
    
    # 1. Build Paley Graph P(37)
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * q
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i] |= 1 << j
                adj[j] |= 1 << i
                edges.append((i, j))
    
    n = q
    
    # Precompute all K4s and independent 4-sets using bitsets
    def compute_k4s_and_ind4s(adj, n):
        k4s = []
        ind4s = []
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        # Check K4
                        if (adj[i] & (1 << j)) and (adj[i] & (1 << k)) and (adj[i] & (1 << l)) and \
                           (adj[j] & (1 << k)) and (adj[j] & (1 << l)) and \
                           (adj[k] & (1 << l)):
                            k4s.append((i, j, k, l))
                        # Check independent 4-set
                        if not (adj[i] & (1 << j)) and not (adj[i] & (1 << k)) and not (adj[i] & (1 << l)) and \
                           not (adj[j] & (1 << k)) and not (adj[j] & (1 << l)) and \
                           not (adj[k] & (1 << l)):
                            ind4s.append((i, j, k, l))
        return k4s, ind4s
    
    k4s, ind4s = compute_k4s_and_ind4s(adj, n)
    
    # Function to check if adding a vertex with neighbor mask 'mask' creates K5 or independent 5
    def count_violations(mask, k4s, ind4s, n):
        viol = 0
        for (a, b, c, d) in k4s:
            # If new vertex connects to all 4, we get K5
            if (mask & (1 << a)) and (mask & (1 << b)) and (mask & (1 << c)) and (mask & (1 << d)):
                viol += 1
        for (a, b, c, d) in ind4s:
            # If new vertex is non-adjacent to all 4, we get independent 5
            if not (mask & (1 << a)) and not (mask & (1 << b)) and not (mask & (1 << c)) and not (mask & (1 << d)):
                viol += 1
        return viol
    
    # Try to find a valid neighbor mask for a new vertex
    def find_valid_mask(n, k4s, ind4s, deadline):
        # Multiple strategies:
        # 1. Start with various biased random assignments
        # 2. Use simulated annealing on the violation count
        
        best_mask = None
        best_viol = float('inf')
        
        num_restarts = 50
        steps_per_restart = 2000
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            # Different initialization strategies
            if restart % 5 == 0:
                # Random 50/50
                mask = random.getrandbits(n)
            elif restart % 5 == 1:
                # Slightly more connections (bias toward avoiding independent 5)
                mask = 0
                for i in range(n):
                    if random.random() < 0.6:
                        mask |= (1 << i)
            elif restart % 5 == 2:
                # Slightly fewer connections (bias toward avoiding K5)
                mask = 0
                for i in range(n):
                    if random.random() < 0.4:
                        mask |= (1 << i)
            elif restart % 5 == 3:
                # Based on degree: connect to high-degree vertices less
                deg = [bin(adj[i]).count('1') for i in range(n)]
                mask = 0
                for i in range(n):
                    # Lower probability for high-degree vertices
                    prob = 0.7 - 0.4 * (deg[i] / n)
                    if random.random() < prob:
                        mask |= (1 << i)
            else:
                # Based on degree: connect to low-degree vertices more
                deg = [bin(adj[i]).count('1') for i in range(n)]
                mask = 0
                for i in range(n):
                    prob = 0.3 + 0.4 * (deg[i] / n)
                    if random.random() < prob:
                        mask |= (1 << i)
            
            viol = count_violations(mask, k4s, ind4s, n)
            if viol == 0:
                return mask
            if viol < best_viol:
                best_viol = viol
                best_mask = mask
            
            # Simulated annealing
            temp = 10.0
            for step in range(steps_per_restart):
                if time.time() > deadline - 5:
                    break
                
                # Pick a random bit to flip
                bit = random.randint(0, n - 1)
                new_mask = mask ^ (1 << bit)
                new_viol = count_violations(new_mask, k4s, ind4s, n)
                
                delta = new_viol - viol
                if delta < 0 or random.random() < (1.0 / (1.0 + delta / temp)):
                    mask = new_mask
                    viol = new_viol
                    if viol == 0:
                        return mask
                    if viol < best_viol:
                        best_viol = viol
                        best_mask = mask
                
                temp *= 0.999
        
        return best_mask if best_viol == 0 else None
    
    # Try to add vertices
    added = 0
    max_add_attempts = 15
    
    for _ in range(max_add_attempts):
        if time.time() > deadline - 30:
            break
        
        mask = find_valid_mask(n, k4s, ind4s, deadline)
        if mask is None:
            break
        
        # Add vertex n with neighbors defined by mask
        new_adj = [0] * (n + 1)
        for i in range(n):
            new_adj[i] = adj[i]
            if mask & (1 << i):
                new_adj[i] |= 1 << n
                new_adj[n] |= 1 << i
                edges.append((i, n))
        adj = new_adj
        n += 1
        added += 1
        
        # Update k4s and ind4s: only need to add those involving the new vertex
        new_k4s = []
        for i in range(n - 1):
            if not (adj[n - 1] & (1 << i)):
                continue
            for j in range(i + 1, n - 1):
                if not (adj[n - 1] & (1 << j)) or not (adj[i] & (1 << j)):
                    continue
                for k in range(j + 1, n - 1):
                    if not (adj[n - 1] & (1 << k)) or not (adj[i] & (1 << k)) or not (adj[j] & (1 << k)):
                        continue
                    new_k4s.append((i, j, k, n - 1))
        k4s.extend(new_k4s)
        
        new_ind4s = []
        for i in range(n - 1):
            if adj[n - 1] & (1 << i):
                continue
            for j in range(i + 1, n - 1):
                if (adj[n - 1] & (1 << j)) or (adj[i] & (1 << j)):
                    continue
                for k in range(j + 1, n - 1):
                    if (adj[n - 1] & (1 << k)) or (adj[i] & (1 << k)) or (adj[j] & (1 << k)):
                        continue
                    new_ind4s.append((i, j, k, n - 1))
        ind4s.extend(new_ind4s)
    
    return n, edges
