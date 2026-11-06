import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    N = 50
    
    # Generate Greenwood-Gleason coloring for n=32
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
        
    # C_seed[i][j] for i,j < 32
    C_seed = [[0]*32 for _ in range(32)]
    for i in range(32):
        for j in range(i + 1, 32):
            c = log[(i % 16) ^ (j % 16)] % 3
            C_seed[i][j] = c
            C_seed[j][i] = c

    def get_energy(C):
        e = 0
        for i in range(N):
            for j in range(i + 1, N):
                c = C[i][j]
                for k in range(j + 1, N):
                    if C[i][k] == c and C[j][k] == c:
                        e += 1
        return e

    best_energy = float('inf')
    best_C = None

    for restart in range(20): # Try multiple restarts
        if time.time() > deadline - 5:
            break
            
        # Initialize C
        C = [[0]*N for _ in range(N)]
        
        # Fill seed part
        for i in range(32):
            for j in range(32):
                C[i][j] = C_seed[i][j]
                
        # Fill rest randomly
        for i in range(N):
            for j in range(i + 1, N):
                if i >= 32 or j >= 32:
                    c = random.randint(0, 3)
                    C[i][j] = c
                    C[j][i] = c
                    
        current_energy = get_energy(C)
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_C = [row[:] for row in C]
            
        if best_energy == 0:
            break

        T = 5.0
        cool_rate = 0.9995
        no_improve_count = 0
        max_no_improve = 5000
        
        while time.time() < deadline - 5:
            if current_energy == 0:
                best_energy = 0
                best_C = [row[:] for row in C]
                break
                
            u, v = random.randint(0, N-1), random.randint(0, N-1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
                
            c_old = C[u][v]
            
            # Calculate cnt
            cnt = [0, 0, 0, 0]
            for w in range(N):
                if w == u or w == v:
                    continue
                cuw = C[u][w]
                cvw = C[v][w]
                if cuw == cvw:
                    cnt[cuw] += 1
                    
            c_new = random.randint(0, 3)
            if c_new == c_old:
                continue
                
            delta_e = cnt[c_new] - cnt[c_old]
            
            if delta_e <= 0 or random.random() < math.exp(-delta_e / T):
                C[u][v] = c_new
                C[v][u] = c_new
                current_energy += delta_e
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_C = [row[:] for row in C]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
                if no_improve_count > max_no_improve:
                    break
                    
            T *= cool_rate
            if T < 0.01:
                T = 0.01

    # Finalize
    if best_energy == 0:
        # Convert best_C to list
        colors = []
        for i in range(N):
            for j in range(i + 1, N):
                colors.append(best_C[i][j])
        return {'n': N, 'colors': colors}
    else:
        # Fallback to n=32
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors.append(C_seed[i][j])
        return {'n': 32, 'colors': colors}
