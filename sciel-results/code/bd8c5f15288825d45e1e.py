"""Combined program: greedy C4-free construction with local search using bitmasks."""
import time
import random

def construct():
    start_time = time.time()
    deadline = 145  # safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def check_c4_free(adj, n):
        """Check C4-freeness: no two vertices share 2+ common neighbors."""
        for i in range(n):
            for j in range(i + 1, n):
                common = adj[i] & adj[j]
                if common.bit_count() >= 2:
                    return False
        return True

    def min_degree(adj, n):
        return min(adj[i].bit_count() for i in range(n))

    def try_construct(n, attempts=50):
        target = n - 39
        if target < 1:
            target = 1
        
        # Start with a slightly denser graph to allow for edge removal
        p = (target + 2) / n 
        if p > 0.5: p = 0.5
        
        best_adj = None
        best_md = -1

        for attempt in range(attempts):
            if elapsed() > deadline - 2:
                break

            # 1. Initialize random graph using bitmasks
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)

            # 2. Repair: Remove edges to break C4s
            changed = True
            max_iter = 100
            for _ in range(max_iter):
                if not changed:
                    break
                changed = False
                for i in range(n):
                    if elapsed() > deadline - 2:
                        break
                    for j in range(i + 1, n):
                        common = adj[i] & adj[j]
                        if common.bit_count() >= 2:
                            # Find two common neighbors
                            c_list = []
                            tmp = common
                            while tmp:
                                lsb = tmp & -tmp
                                idx = lsb.bit_length() - 1
                                c_list.append(idx)
                                tmp ^= lsb
                            
                            w = c_list[0]
                            # Remove edge (i, w) to break the C4
                            adj[i] &= ~(1 << w)
                            adj[w] &= ~(1 << i)
                            changed = True
                
                if elapsed() > deadline - 2:
                    break

            # 3. Verify C4-free
            if not check_c4_free(adj, n):
                continue

            md = min_degree(adj, n)
            if md > best_md:
                best_md = md
                best_adj = adj[:]

            # 4. If min degree is too low, try to add edges
            if md < target:
                improved = True
                max_local = 20
                for _ in range(max_local):
                    if not improved:
                        break
                    improved = False
                    if elapsed() > deadline - 2:
                        break
                    
                    degs = [adj[i].bit_count() for i in range(n)]
                    min_deg = min(degs)
                    if min_deg >= target:
                        break
                    
                    low_verts = [i for i in range(n) if degs[i] <= min_deg + 1]
                    random.shuffle(low_verts)
                    
                    for u in low_verts:
                        if degs[u] >= target:
                            continue
                        mask = (1 << n) - 1
                        non_neighbors = (~adj[u]) & mask
                        non_neighbors &= ~(1 << u)
                        
                        nn_list = []
                        tmp = non_neighbors
                        while tmp:
                            lsb = tmp & -tmp
                            idx = lsb.bit_length() - 1
                            nn_list.append(idx)
                            tmp ^= lsb
                        
                        random.shuffle(nn_list)
                        
                        for v in nn_list:
                            common = adj[u] & adj[v]
                            if common.bit_count() < 2:
                                adj[u] |= (1 << v)
                                adj[v] |= (1 << u)
                                degs[u] += 1
                                degs[v] += 1
                                improved = True
                                if degs[u] >= target:
                                    break

            # Final check
            if check_c4_free(adj, n):
                md = min_degree(adj, n)
                if md > best_md:
                    best_md = md
                    best_adj = adj[:]
                if md >= target:
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i] & (1 << j):
                                edges.append([i, j])
                    return n, edges

        if best_adj is not None and best_md >= target:
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if best_adj[i] & (1 << j):
                        edges.append([i, j])
            return n, edges
            
        return None

    # Try from largest n down
    for n in [47, 46, 45, 44, 43, 42, 41, 40]:
        result = try_construct(n)
        if result is not None:
            return {'n': result[0], 'edges': result[1]}

    # Fallback: trivial graph
    return {'n': 39, 'edges': []}
