import time
import random

def popcount(x):
    return x.bit_count() if hasattr(int, 'bit_count') else bin(x).count('1')

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 20
    best_edges = []
    best_count = 0
    
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    while time.time() < deadline:
        random.shuffle(all_edges)
        
        adj = [0] * n
        edge_list = []
        
        for u, v in all_edges:
            common_uv = adj[u] & adj[v]
            
            if popcount(common_uv) < 2:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_list.append([u, v])
                continue
            
            valid = True
            temp = common_uv
            bits = []
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                bits.append(idx)
                temp ^= lsb
            
            for i in range(len(bits)):
                r1 = bits[i]
                for j in range(i+1, len(bits)):
                    r2 = bits[j]
                    common_v_r1_r2 = adj[v] & adj[r1] & adj[r2]
                    common_v_r1_r2 &= ~(1 << u)
                    if popcount(common_v_r1_r2) >= 2:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_list.append([u, v])
        
        if len(edge_list) > best_count:
            best_count = len(edge_list)
            best_edges = edge_list[:]
            
    return best_edges
