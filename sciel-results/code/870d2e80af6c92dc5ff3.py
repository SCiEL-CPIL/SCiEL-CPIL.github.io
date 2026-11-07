# EVOLVE-BLOCK-START
import numpy as np
from datetime import datetime, timedelta

def sign_edges(graph, budget):
    start_time = datetime.now()
    deadline = start_time + timedelta(seconds=80)
    
    m = graph['n'] // 2
    edges = graph['edges']
    
    # Classify each edge
    edge_info = {}
    for u, v in edges:
        if u < m and v < m:
            i = u % m
            edge_info[(u, v)] = ('a', i)
            edge_info[(v, u)] = ('a', i)
        elif u >= m and v >= m:
            i = (u - m) % m
            edge_info[(u, v)] = ('b', i)
            edge_info[(v, u)] = ('b', i)
        else:
            i = min(u, v) % m
            edge_info[(u, v)] = ('c', i)
            edge_info[(v, u)] = ('c', i)
    
    def build_A(a, b, c):
        n = 2 * m
        A = np.zeros((n, n))
        for u, v in edges:
            etype, idx = edge_info[(u, v)]
            if etype == 'a':
                s = a[idx]
            elif etype == 'b':
                s = b[idx]
            else:
                s = c[idx]
            A[u, v] = s
            A[v, u] = s
        return A
    
    best_radius = float('inf')
    best_signs = None
    
    for a_mask in range(2 ** (m - 1)):
        if datetime.now() > deadline:
            break
        a = np.array([1 if (a_mask >> i) & 1 else -1 for i in range(m - 1)])
        a = np.append(a, np.prod(a))
        
        for b_mask in range(2 ** (m - 1)):
            if datetime.now() > deadline:
                break
            b = np.array([1 if (b_mask >> i) & 1 else -1 for i in range(m - 1)])
            b = np.append(b, np.prod(b))
            
            for c0 in (1, -1):
                if datetime.now() > deadline:
                    break
                c = np.empty(m)
                c[0] = c0
                rp = 1
                for i in range(1, m):
                    rp *= a[i-1] * b[i-1]
                    c[i] = c0 * rp
                
                A = build_A(a, b, c)
                r = np.max(np.abs(np.linalg.eigvalsh(A)))
                if r < best_radius:
                    best_radius = r
                    signs = np.empty(len(edges))
                    for idx, (u, v) in enumerate(edges):
                        etype, ei = edge_info[(u, v)]
                        if etype == 'a':
                            signs[idx] = a[ei]
                        elif etype == 'b':
                            signs[idx] = b[ei]
                        else:
                            signs[idx] = c[ei]
                    best_signs = signs
    
    if best_signs is None:
        best_signs = np.ones(len(edges))
    
    return best_signs
# EVOLVE-BLOCK-END
