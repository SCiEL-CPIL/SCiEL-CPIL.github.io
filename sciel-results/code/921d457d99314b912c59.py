import random
import time
import numpy as np

def construct():
    n = 20
    deadline = time.time() + 105  # Safety margin
    
    best_edges = 0
    best_adj = [[0]*n for _ in range(n)]
    
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def build_bitmask(adj):
        bm = [0] * n
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    bm[u] |= (1 << v)
                    bm[v] |= (1 << u)
        return bm

    def count_edges(bm):
        return sum(bin(x).count('1') for x in bm) // 2

    def is_valid(bm):
        """Check if graph is K3,3-free using bitmasks."""
        for i in range(n):
            for j in range(i+1, n):
                common_ij = bm[i] & bm[j]
                # Iterate over bits in common_ij
                c = common_ij
                while c:
                    lsb = c & -c
                    k = lsb.bit_length() - 1
                    c -= lsb
                    if k <= j:
                        continue
                    # Check triple (i, j, k)
                    common_ijk = common_ij & bm[k]
                    if bin(common_ijk).count('1') >= 3:
                        return False
        return True

    def penalty(bm):
        """Calculate penalty for K3,3 violations."""
        p = 0
        for i in range(n):
            for j in range(i+1, n):
                common_ij = bm[i] & bm[j]
                c = common_ij
                while c:
                    lsb = c & -c
                    k = lsb.bit_length() - 1
                    c -= lsb
                    if k <= j:
                        continue
                    common_ijk = common_ij & bm[k]
                    cnt = bin(common_ijk).count('1')
                    if cnt >= 3:
                        p += (cnt - 2) ** 2
        return p

    def can_add_fast(u, v, bm):
        """Fast check if adding (u,v) creates K3,3."""
        # Tentatively add
        bm[u] |= (1 << v)
        bm[v] |= (1 << u)
        
        valid = True
        # Check triples involving u
        for x in range(n):
            if x == u or x == v: continue
            for y in range(x+1, n):
                if y == u or y == v: continue
                # Triple (u, x, y)
                common = bm[u] & bm[x] & bm[y]
                if bin(common).count('1') >= 3:
                    valid = False
                    break
            if not valid: break
        if valid:
            # Check triples involving v
            for x in range(n):
                if x == u or x == v: continue
                for y in range(x+1, n):
                    if y == u or y == v: continue
                    # Triple (v, x, y)
                    common = bm[v] & bm[x] & bm[y]
                    if bin(common).count('1') >= 3:
                        valid = False
                        break
                if not valid: break
        
        # Revert
        bm[u] &= ~(1 << v)
        bm[v] &= ~(1 << u)
        
        return valid

    def greedy_fill(bm):
        """Greedy add missing edges."""
        improved = True
        while improved:
            improved = False
            for u, v in all_pairs:
                if not (bm[u] >> v) & 1:
                    if can_add_fast(u, v, bm):
                        bm[u] |= (1 << v)
                        bm[v] |= (1 << u)
                        improved = True
        return bm

    def simulated_annealing(bm, time_limit):
        """Simulated annealing to improve graph."""
        start_time = time.time()
        T = 5.0
        T_min = 0.01
        cooling = 0.995
        
        current_energy = -count_edges(bm) + penalty(bm)
        best_bm = bm[:]
        best_energy = current_energy
        
        while time.time() - start_time < time_limit:
            # Pick a random edge to flip
            u, v = random.choice(all_pairs)
            
            # Flip
            if (bm[u] >> v) & 1:
                bm[u] &= ~(1 << v)
                bm[v] &= ~(1 << u)
            else:
                bm[u] |= (1 << v)
                bm[v] |= (1 << u)
            
            new_energy = -count_edges(bm) + penalty(bm)
            delta = new_energy - current_energy
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_bm = bm[:]
            else:
                # Revert
                if (bm[u] >> v) & 1:
                    bm[u] &= ~(1 << v)
                    bm[v] &= ~(1 << u)
                else:
                    bm[u] |= (1 << v)
                    bm[v] |= (1 << u)
            
            T = max(T * cooling, T_min)
            
            # Periodic restart of temperature if stuck
            if T == T_min:
                if random.random() < 0.1:
                    T = 5.0 * random.random()
        
        return best_bm

    # Main loop
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        
        # Random initialization
        p = 0.5 + 0.2 * (restart_count % 3) / 2.0
        bm = [0] * n
        for u, v in all_pairs:
            if random.random() < p:
                bm[u] |= (1 << v)
                bm[v] |= (1 << u)
        
        # Purge: remove edges to make valid
        while not is_valid(bm):
            # Find a violation
            for i in range(n):
                for j in range(i+1, n):
                    common_ij = bm[i] & bm[j]
                    c = common_ij
                    found = False
                    while c:
                        lsb = c & -c
                        k = lsb.bit_length() - 1
                        c -= lsb
                        if k <= j:
                            continue
                        common_ijk = common_ij & bm[k]
                        if bin(common_ijk).count('1') >= 3:
                            # Remove a random edge from the K3,3
                            # Pick random x from common_ijk
                            x_list = []
                            cc = common_ijk
                            while cc:
                                lsb2 = cc & -cc
                                x_list.append(lsb2.bit_length() - 1)
                                cc -= lsb2
                            x = random.choice(x_list)
                            # Remove edge (i, x)
                            bm[i] &= ~(1 << x)
                            bm[x] &= ~(1 << i)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
        
        # Greedy fill
        bm = greedy_fill(bm)
        
        # Simulated Annealing
        remaining = deadline - time.time()
        if remaining > 5:
            sa_time = min(2.0, remaining * 0.3)
            bm = simulated_annealing(bm, sa_time)
        
        # Final greedy fill
        bm = greedy_fill(bm)
        
        # Update best
        edges = count_edges(bm)
        if edges > best_edges:
            best_edges = edges
            # Convert bm to adj
            best_adj = [[0]*n for _ in range(n)]
            for u in range(n):
                for v in range(u+1, n):
                    if (bm[u] >> v) & 1:
                        best_adj[u][v] = 1
                        best_adj[v][u] = 1
        
        if best_edges >= 85:
            break

    # Final polish on best_adj
    if time.time() < deadline - 2:
        best_bm = build_bitmask(best_adj)
        # Try to add any remaining edges
        for u, v in all_pairs:
            if not (best_bm[u] >> v) & 1:
                if can_add_fast(u, v, best_bm):
                    best_bm[u] |= (1 << v)
                    best_bm[v] |= (1 << u)
        
        # Convert back
        best_adj = [[0]*n for _ in range(n)]
        for u in range(n):
            for v in range(u+1, n):
                if (best_bm[u] >> v) & 1:
                    best_adj[u][v] = 1
                    best_adj[v][u] = 1

    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                result.append([i, j])
    return result
