import random
import time
import math

def construct():
    n = 16
    # Initialize K_8,8 + (0,1)
    A = [[0]*n for _ in range(n)]
    for u in range(8):
        for v in range(8, 16):
            A[u][v] = 1
            A[v][u] = 1
    A[0][1] = 1
    A[1][0] = 1
    
    # Use a set for fast edge lookup and a list for removal selection
    edges_set = set()
    for u in range(8):
        for v in range(8, 16):
            edges_set.add((u, v))
    edges_set.add((0, 1))
    
    # List of edges for removal selection (indices correspond to edges_set contents)
    edge_list = list(edges_set)
    
    # Pre-compute all possible pairs for faster random selection
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    non_edges_set = set(all_pairs) - edges_set
    
    def compute_degs():
        degs = [0]*n
        for u, v in edges_set:
            degs[u] += 1
            degs[v] += 1
        return degs
    
    def get_score(degs=None):
        if degs is None:
            degs = compute_degs()
        
        max_sum = 0
        # Iterate over all triangles
        for i in range(n):
            for j in range(i+1, n):
                if A[i][j]:
                    for k in range(j+1, n):
                        if A[i][k] and A[j][k]:
                            s = degs[i] + degs[j] + degs[k]
                            if s > max_sum:
                                max_sum = s
        return max_sum
    
    def find_worst_triangles(degs):
        """Find triangles with degree sum close to the maximum."""
        tri_sums = []
        for i in range(n):
            for j in range(i+1, n):
                if A[i][j]:
                    for k in range(j+1, n):
                        if A[i][k] and A[j][k]:
                            s = degs[i] + degs[j] + degs[k]
                            tri_sums.append((s, (i, j, k)))
        tri_sums.sort(reverse=True)
        return tri_sums
    
    current_score = get_score()
    best_score = current_score
    best_A = [row[:] for row in A]
    best_degs = compute_degs()
    
    start_time = time.time()
    time_limit = 105.0  # Leave more margin for polish pass
    
    T = 10.0
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        # 2-Swap Move
        idx1 = random.randint(0, len(edge_list) - 1)
        idx2 = random.randint(0, len(edge_list) - 1)
        while idx2 == idx1:
            idx2 = random.randint(0, len(edge_list) - 1)
        
        e1 = edge_list[idx1]
        e2 = edge_list[idx2]
        
        # Pick 2 distinct non-edges to add
        new_e1 = None
        new_e2 = None
        
        while True:
            p1 = all_pairs[random.randint(0, len(all_pairs)-1)]
            if p1 not in edges_set:
                new_e1 = p1
                break
            
        while True:
            p2 = all_pairs[random.randint(0, len(all_pairs)-1)]
            if p2 not in edges_set and p2 != new_e1:
                new_e2 = p2
                break
        
        # Perform move
        edges_set.remove(e1)
        edges_set.remove(e2)
        A[e1[0]][e1[1]] = 0
        A[e1[1]][e1[0]] = 0
        A[e2[0]][e2[1]] = 0
        A[e2[1]][e2[0]] = 0
        
        edges_set.add(new_e1)
        edges_set.add(new_e2)
        A[new_e1[0]][new_e1[1]] = 1
        A[new_e1[1]][new_e1[0]] = 1
        A[new_e2[0]][new_e2[1]] = 1
        A[new_e2[1]][new_e2[0]] = 1
        
        edge_list[idx1] = new_e1
        edge_list[idx2] = new_e2
        
        # Compute new degrees and score
        new_degs = compute_degs()
        new_score = get_score(new_degs)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_A = [row[:] for row in A]
                best_degs = new_degs[:]
        else:
            delta = new_score - current_score
            if T > 0.1:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_score = new_score
                else:
                    # Revert
                    edges_set.discard(new_e1)
                    edges_set.discard(new_e2)
                    edges_set.add(e1)
                    edges_set.add(e2)
                    
                    A[new_e1[0]][new_e1[1]] = 0
                    A[new_e1[1]][new_e1[0]] = 0
                    A[new_e2[0]][new_e2[1]] = 0
                    A[new_e2[1]][new_e2[0]] = 0
                    
                    A[e1[0]][e1[1]] = 1
                    A[e1[1]][e1[0]] = 1
                    A[e2[0]][e2[1]] = 1
                    A[e2[1]][e2[0]] = 1
                    
                    edge_list[idx1] = e1
                    edge_list[idx2] = e2
            else:
                # Revert
                edges_set.discard(new_e1)
                edges_set.discard(new_e2)
                edges_set.add(e1)
                edges_set.add(e2)
                
                A[new_e1[0]][new_e1[1]] = 0
                A[new_e1[1]][new_e1[0]] = 0
                A[new_e2[0]][new_e2[1]] = 0
                A[new_e2[1]][new_e2[0]] = 0
                
                A[e1[0]][e1[1]] = 1
                A[e1[1]][e1[0]] = 1
                A[e2[0]][e2[1]] = 1
                A[e2[1]][e2[0]] = 1
                
                edge_list[idx1] = e1
                edge_list[idx2] = e2
        
        T *= cooling
    
    # === Final polish pass: targeted local search on best solution ===
    # Restore best solution
    A = [row[:] for row in best_A]
    edges_set = set()
    for i in range(n):
        for j in range(i+1, n):
            if A[i][j]:
                edges_set.add((i, j))
    
    cur_degs = compute_degs()
    cur_score = get_score(cur_degs)
    
    polish_deadline = time.time() + 40.0  # Use remaining time for polish
    
    # Polish: try to reduce max triangle degree sum via targeted swaps
    improved = True
    while improved and time.time() < polish_deadline:
        improved = False
        degs = compute_degs()
        worst_tris = find_worst_triangles(degs)
        
        if not worst_tris:
            break
        
        max_tri_sum = worst_tris[0][0]
        
        # For each worst triangle, try swaps that might reduce its degree sum
        for tri_sum, tri in worst_tris[:5]:  # Only consider top 5 worst triangles
            if tri_sum < max_tri_sum - 1:
                break
            
            i, j, k = tri
            # Try removing one edge from the triangle and adding a non-edge
            # that doesn't create a worse triangle
            for rem_edge in [(i,j), (i,k), (j,k)]:
                if rem_edge not in edges_set:
                    continue
                
                # Find non-edges to add
                candidates = []
                for p in all_pairs:
                    if p not in edges_set and p != rem_edge:
                        candidates.append(p)
                
                if not candidates:
                    continue
                
                # Try a few random candidates
                random.shuffle(candidates)
                for new_edge in candidates[:20]:
                    # Test the swap
                    edges_set.remove(rem_edge)
                    edges_set.add(new_edge)
                    
                    A[rem_edge[0]][rem_edge[1]] = 0
                    A[rem_edge[1]][rem_edge[0]] = 0
                    A[new_edge[0]][new_edge[1]] = 1
                    A[new_edge[1]][new_edge[0]] = 1
                    
                    test_degs = compute_degs()
                    test_score = get_score(test_degs)
                    
                    if test_score < cur_score:
                        cur_score = test_score
                        cur_degs = test_degs
                        improved = True
                        break
                    else:
                        # Revert
                        edges_set.discard(new_edge)
                        edges_set.add(rem_edge)
                        A[new_edge[0]][new_edge[1]] = 0
                        A[new_edge[1]][new_edge[0]] = 0
                        A[rem_edge[0]][rem_edge[1]] = 1
                        A[rem_edge[1]][rem_edge[0]] = 1
                
                if improved:
                    break
            
            if improved:
                break
        
        if time.time() >= polish_deadline:
            break
    
    # Reconstruct edges from final A
    final_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if A[i][j]:
                final_edges.append([i, j])
                
    return {'n': n, 'edges': final_edges}
