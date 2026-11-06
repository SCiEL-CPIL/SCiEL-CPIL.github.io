import numpy as np
import time
import random
from collections import deque

def construct():
    N = 51975
    # Get divisors > 1
    divisors = []
    for i in range(2, int(np.sqrt(N)) + 1):
        if N % i == 0:
            divisors.append(i)
            if i != N // i:
                divisors.append(N // i)
    if N not in divisors:
        divisors.append(N)
    
    divisors.sort()
    
    start_time = time.time()
    timeout = 120.0
    
    # Helper to calculate score
    def calc_score(classes):
        if not classes:
            return N
        covered = np.zeros(N, dtype=np.uint8)
        for m, r in classes:
            covered[r::m] = 1
        return int(np.sum(1 - covered))
    
    # Maximum flow using Ford-Fulkerson with BFS (Edmonds-Karp)
    # Source -> Moduli (capacity 1) -> Residues (capacity 1) -> Sink (capacity 1)
    def max_flow_bipartite(divisors, N, timeout, start_time):
        """
        Build bipartite graph: moduli on left, residues on right
        Edge (m, r) exists if r < m (all residues for each modulus)
        Find maximum matching
        """
        # Node indexing:
        # 0: source
        # 1..M: moduli (M = len(divisors))
        # M+1..M+N: residues
        # M+N+1: sink
        M = len(divisors)
        num_nodes = M + N + 2
        source = 0
        sink = num_nodes - 1
        
        # Adjacency list for residual graph
        # adj[u] = list of (v, capacity)
        adj = [[] for _ in range(num_nodes)]
        
        def add_edge(u, v, cap):
            adj[u].append([v, cap])
            adj[v].append([u, 0])  # Reverse edge with 0 capacity
        
        # Source to moduli
        for i, m in enumerate(divisors):
            add_edge(source, 1 + i, 1)
        
        # Moduli to residues (edge if residue r < m)
        for i, m in enumerate(divisors):
            for r in range(m):
                add_edge(1 + i, 1 + M + r, 1)
        
        # Residues to sink
        for r in range(N):
            add_edge(1 + M + r, sink, 1)
        
        # BFS to find augmenting path
        def bfs():
            parent = [-1] * num_nodes
            parent[source] = source
            queue = deque([source])
            
            while queue:
                u = queue.popleft()
                for v, cap in adj[u]:
                    if cap > 0 and parent[v] == -1:
                        parent[v] = u
                        if v == sink:
                            return parent
                        queue.append(v)
            return None
        
        flow = 0
        while time.time() - start_time < timeout - 5:
            parent = bfs()
            if parent is None:
                break
            
            # Find minimum capacity along the path
            min_cap = float('inf')
            v = sink
            while v != source:
                u = parent[v]
                # Find the edge from u to v
                for edge in adj[u]:
                    if edge[0] == v and edge[1] > 0:
                        min_cap = min(min_cap, edge[1])
                        break
                v = u
            
            # Update residual capacities
            v = sink
            while v != source:
                u = parent[v]
                for i, edge in enumerate(adj[u]):
                    if edge[0] == v and edge[1] > 0:
                        adj[u][i][1] -= min_cap
                        # Find reverse edge
                        for j, rev_edge in enumerate(adj[v]):
                            if rev_edge[0] == u:
                                adj[v][j][1] += min_cap
                                break
                        break
                v = u
            
            flow += min_cap
        
        # Extract matching from residual graph
        matching = {}  # modulus_index -> residue
        for i, m in enumerate(divisors):
            for v, cap in adj[1 + i]:
                if cap < 0:  # Reverse edge with positive flow
                    r = v - 1 - M
                    matching[i] = r
                    break
        
        return matching, flow
    
    # Greedy construction for initial solution
    def greedy_construction(divisors, start_time, timeout, randomize_order=False):
        covered = np.zeros(N, dtype=np.uint8)
        used_moduli = set()
        classes = []
        divs = list(divisors)
        if randomize_order:
            random.shuffle(divs)
            
        while time.time() - start_time < timeout - 5:
            best_gain = 0
            best_m = None
            best_r = None
            
            for m in divs:
                if m in used_moduli:
                    continue
                
                max_r_gain = 0
                max_r_val = -1
                
                for r in range(m):
                    gain = np.sum(1 - covered[r::m])
                    if gain > max_r_gain:
                        max_r_gain = gain
                        max_r_val = r
                
                if max_r_gain > best_gain:
                    best_gain = max_r_gain
                    best_m = m
                    best_r = max_r_val
            
            if best_gain == 0:
                break
                
            classes.append([best_m, best_r])
            used_moduli.add(best_m)
            covered[best_r::best_m] = 1
            
        return classes
    
    # Local search
    def local_search(classes, start_time, timeout):
        current_score = calc_score(classes)
        improved = True
        classes = [c[:] for c in classes]
        
        while improved and time.time() - start_time < timeout - 2:
            improved = False
            used_moduli = set(m for m, r in classes)
            unused_moduli = [m for m in divisors if m not in used_moduli]
            
            for i in range(len(classes)):
                if time.time() - start_time > timeout - 2:
                    break
                
                m_old, r_old = classes[i]
                
                for m_new in unused_moduli:
                    if time.time() - start_time > timeout - 2:
                        break
                    
                    covered_no_old = np.zeros(N, dtype=np.uint8)
                    for j, (m_j, r_j) in enumerate(classes):
                        if j != i:
                            covered_no_old[r_j::m_j] = 1
                    
                    target = 1 - covered_no_old
                    best_r_new = -1
                    best_gain_new = 0
                    
                    for r_new in range(m_new):
                        gain = np.sum(target[r_new::m_new])
                        if gain > best_gain_new:
                            best_gain_new = gain
                            best_r_new = r_new
                    
                    if best_r_new != -1:
                        old_unique = np.sum((1 - covered_no_old)[r_old::m_old])
                        
                        if best_gain_new > old_unique:
                            classes[i] = [m_new, best_r_new]
                            current_score -= (old_unique - best_gain_new)
                            improved = True
                            break
                if improved:
                    break
            
        return classes, current_score
    
    # Main: Try max flow first, then greedy + local search
    best_classes = []
    best_score = N
    
    # Phase 1: Maximum flow approach
    if time.time() - start_time < timeout - 30:
        try:
            matching, flow = max_flow_bipartite(divisors, N, timeout, start_time)
            # Convert matching to classes
            flow_classes = []
            for i, m in enumerate(divisors):
                if i in matching:
                    r = matching[i]
                    flow_classes.append([m, r])
            
            if flow_classes:
                flow_score = calc_score(flow_classes)
                if flow_score < best_score:
                    best_score = flow_score
                    best_classes = [c[:] for c in flow_classes]
        except Exception:
            pass
    
    # Phase 2: Greedy + Local Search with random restarts
    num_restarts = 0
    while time.time() - start_time < timeout - 10:
        num_restarts += 1
        is_random = (num_restarts > 1)
        
        greedy_classes = greedy_construction(divisors, start_time, timeout, randomize_order=is_random)
        
        ls_classes, ls_score = local_search(greedy_classes, start_time, timeout)
        
        if ls_score < best_score:
            best_score = ls_score
            best_classes = [c[:] for c in ls_classes]
        
        if best_score == 0:
            break
    
    return best_classes
