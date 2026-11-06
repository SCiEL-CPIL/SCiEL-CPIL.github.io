import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    best_edges = []
    best_count = 0
    
    # adj[u] is an int where bit v is 1 if (u,v) is an edge
    adj = [0] * n

    def bit_list(mask):
        return [i for i in range(n) if (mask >> i) & 1]

    def creates_q3(adj, u, v):
        # Check if adding edge (u,v) creates a Q3.
        # Assumes (u,v) is NOT in adj.
        
        # Temporarily add
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        found = False
        nu = adj[u] & ~(1 << v)
        nv = adj[v] & ~(1 << u)
        
        nu_list = bit_list(nu)
        nv_list = bit_list(nv)
        
        for i in range(len(nu_list)):
            w1 = nu_list[i]
            for j in range(i+1, len(nu_list)):
                w2 = nu_list[j]
                
                y1_mask = (adj[w1] & adj[w2]) & ~(1 << u) & ~(1 << v)
                if y1_mask == 0: continue
                
                for k in range(len(nv_list)):
                    z1 = nv_list[k]
                    for l in range(k+1, len(nv_list)):
                        z2 = nv_list[l]
                        
                        if w1 == z1 or w1 == z2 or w2 == z1 or w2 == z2: continue
                        
                        y2_mask = (adj[z1] & adj[z2]) & ~(1 << u) & ~(1 << v)
                        if y2_mask == 0: continue
                        
                        # Check connection patterns
                        # Pattern 1: w1-z1, w2-z2
                        if (adj[w1] >> z1) & 1 and (adj[w2] >> z2) & 1:
                            # Check y1-y2
                            y1_iter = y1_mask
                            while y1_iter:
                                y1 = (y1_iter & -y1_iter).bit_length() - 1
                                y1_iter &= y1_iter - 1
                                if y2_mask & ~(1 << y1) & adj[y1]:
                                    found = True
                                    break
                            if found: break
                        if found: break
                        
                        # Pattern 2: w1-z2, w2-z1
                        if (adj[w1] >> z2) & 1 and (adj[w2] >> z1) & 1:
                            y1_iter = y1_mask
                            while y1_iter:
                                y1 = (y1_iter & -y1_iter).bit_length() - 1
                                y1_iter &= y1_iter - 1
                                if y2_mask & ~(1 << y1) & adj[y1]:
                                    found = True
                                    break
                            if found: break
                        if found: break
                    if found: break
                if found: break
            if found: break
            
        # Revert
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
        return found

    def find_q3(adj):
        # Find any Q3 in the graph. Returns list of edges or None.
        # Iterate edges
        for u in range(n):
            for v in range(u+1, n):
                if not ((adj[u] >> v) & 1):
                    continue
                
                # Check if (u,v) is in a Q3
                # Similar to creates_q3, but (u,v) is present.
                # We need to find w1,w2,z1,z2,y1,y2.
                
                nu = adj[u] & ~(1 << v)
                nv = adj[v] & ~(1 << u)
                
                nu_list = bit_list(nu)
                nv_list = bit_list(nv)
                
                for i in range(len(nu_list)):
                    w1 = nu_list[i]
                    for j in range(i+1, len(nu_list)):
                        w2 = nu_list[j]
                        
                        y1_mask = (adj[w1] & adj[w2]) & ~(1 << u) & ~(1 << v)
                        if y1_mask == 0: continue
                        
                        for k in range(len(nv_list)):
                            z1 = nv_list[k]
                            for l in range(k+1, len(nv_list)):
                                z2 = nv_list[l]
                                
                                if w1 == z1 or w1 == z2 or w2 == z1 or w2 == z2: continue
                                
                                y2_mask = (adj[z1] & adj[z2]) & ~(1 << u) & ~(1 << v)
                                if y2_mask == 0: continue
                                
                                # Pattern 1
                                if (adj[w1] >> z1) & 1 and (adj[w2] >> z2) & 1:
                                    y1_iter = y1_mask
                                    while y1_iter:
                                        y1 = (y1_iter & -y1_iter).bit_length() - 1
                                        y1_iter &= y1_iter - 1
                                        y2_cand = y2_mask & ~(1 << y1) & adj[y1]
                                        if y2_cand:
                                            y2 = (y2_cand & -y2_cand).bit_length() - 1
                                            # Return edges of this Q3
                                            return [(u,v), (u,w1), (u,w2), (v,z1), (v,z2),
                                                    (w1,z1), (w2,z2), (w1,y1), (w2,y1),
                                                    (z1,y2), (z2,y2), (y1,y2)]
                                # Pattern 2
                                if (adj[w1] >> z2) & 1 and (adj[w2] >> z1) & 1:
                                    y1_iter = y1_mask
                                    while y1_iter:
                                        y1 = (y1_iter & -y1_iter).bit_length() - 1
                                        y1_iter &= y1_iter - 1
                                        y2_cand = y2_mask & ~(1 << y1) & adj[y1]
                                        if y2_cand:
                                            y2 = (y2_cand & -y2_cand).bit_length() - 1
                                            return [(u,v), (u,w1), (u,w2), (v,z1), (v,z2),
                                                    (w1,z2), (w2,z1), (w1,y1), (w2,y1),
                                                    (z1,y2), (z2,y2), (y1,y2)]
        return None

    def edge_count(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def run_search(seed):
        random.seed(seed)
        # Initialize with random graph
        adj_local = [0] * n
        for u, v in all_edges:
            if random.random() < 0.6:
                adj_local[u] |= (1 << v)
                adj_local[v] |= (1 << u)
        
        # Repair
        while True:
            q3 = find_q3(adj_local)
            if q3 is None:
                break
            # Remove random edge from q3
            e = random.choice(q3)
            u, v = e
            adj_local[u] &= ~(1 << v)
            adj_local[v] &= ~(1 << u)
            
        # Optimize
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try adding edges
            for u, v in all_edges:
                if (adj_local[u] >> v) & 1:
                    continue
                if not creates_q3(adj_local, u, v):
                    adj_local[u] |= (1 << v)
                    adj_local[v] |= (1 << u)
                    improved = True
            
            # Try swaps (1 out, 2 in)
            # This is complex to implement efficiently in Python.
            # Let's stick to add/remove and maybe restarts.
            # Actually, "remove one, add two" is powerful.
            # But let's see if greedy add is enough with multiple restarts.
            
            # If no improvement, try random restarts inside?
            # No, the outer loop handles restarts.
            
            # To avoid getting stuck, we can try removing a random edge and re-optimizing?
            # That's simulated annealing.
            
            # Let's add a simple SA step:
            if not improved:
                # Pick random edge to remove
                if edge_count(adj_local) > 0:
                    # Find an edge
                    edges_present = [(u, v) for u, v in all_edges if (adj_local[u] >> v) & 1]
                    if edges_present:
                        u, v = random.choice(edges_present)
                        adj_local[u] &= ~(1 << v)
                        adj_local[v] &= ~(1 << u)
                        # Try to add 2 edges
                        # This is getting complicated. Let's just rely on restarts.
                        pass

        return edge_count(adj_local), [(u, v) for u, v in all_edges if (adj_local[u] >> v) & 1]

    restart = 0
    while time.time() < deadline:
        count, edges_list = run_search(restart)
        if count > best_count:
            best_count = count
            best_edges = edges_list
        
        restart += 1
        if best_count >= 52:
             # High score, break early? No, keep trying.
             pass

    return [[u, v] for u, v in best_edges]

if __name__ == "__main__":
    print(construct())
