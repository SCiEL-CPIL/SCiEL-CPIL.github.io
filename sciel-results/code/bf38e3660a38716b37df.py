import random
import time

def construct():
    # Fallback: Seed solution for n=32
    def get_seed_solution():
        def mul(a, b):
            r = 0
            for i in range(4):
                if b >> i & 1:
                    r ^= a << i
            for i in range(7, 3, -1):
                if r >> i & 1:
                    r ^= 0b10011 << (i - 4)
            return r
        log, x = {}, 1
        for e in range(15):
            log[x] = e
            x = mul(x, 2)
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
        return {'n': 32, 'colors': colors}

    # Try n=50
    n = 50
    start_time = time.time()
    max_time = 115
    
    # Data structures
    col = [[0]*n for _ in range(n)]
    N = [ [set() for _ in range(n)] for _ in range(4) ]
    tri = [0]*4
    
    for i in range(n):
        for j in range(i+1, n):
            c = random.randint(0, 3)
            col[i][j] = c
            col[j][i] = c
            N[c][i].add(j)
            N[c][j].add(i)
            
    # Init tri counts
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                c = col[i][j]
                if col[i][k] == c and col[j][k] == c:
                    tri[c] += 1
                    
    best_tri = sum(tri)
    
    # Local Search
    while time.time() - start_time < max_time:
        u = random.randint(0, n-1)
        v = random.randint(u+1, n-1)
        old_c = col[u][v]
        
        best_delta = 0
        best_new_c = old_c
        best_destroyed = 0
        best_created = 0
        
        for new_c in range(4):
            if new_c == old_c:
                continue
            
            # Calculate destroyed (triangles of old_c broken)
            set_u_old = N[old_c][u]
            set_v_old = N[old_c][v]
            destroyed = 0
            if len(set_u_old) < len(set_v_old):
                for w in set_u_old:
                    if w in set_v_old:
                        destroyed += 1
            else:
                for w in set_v_old:
                    if w in set_u_old:
                        destroyed += 1
                        
            # Calculate created (triangles of new_c formed)
            set_u_new = N[new_c][u]
            set_v_new = N[new_c][v]
            created = 0
            if len(set_u_new) < len(set_v_new):
                for w in set_u_new:
                    if w in set_v_new:
                        created += 1
            else:
                for w in set_v_new:
                    if w in set_u_new:
                        created += 1
                        
            delta = created - destroyed
            if delta < best_delta:
                best_delta = delta
                best_new_c = new_c
                best_destroyed = destroyed
                best_created = created
                
        if best_new_c != old_c:
            # Apply
            N[old_c][u].remove(v)
            N[old_c][v].remove(u)
            N[best_new_c][u].add(v)
            N[best_new_c][v].add(u)
            col[u][v] = best_new_c
            col[v][u] = best_new_c
            
            tri[old_c] -= best_destroyed
            tri[best_new_c] += best_created
            
            current_tri = sum(tri)
            if current_tri < best_tri:
                best_tri = current_tri
                
            if best_tri == 0:
                break
        else:
            # No improvement, maybe restart?
            # For simplicity, just continue.
            pass
            
    if best_tri == 0:
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                colors.append(col[i][j])
        return {'n': n, 'colors': colors}
    else:
        return get_seed_solution()
