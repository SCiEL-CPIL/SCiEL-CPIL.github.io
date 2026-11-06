import time

def construct():
            N = 40
            start_time = time.time()
            deadline = start_time + 50 # Safety margin

            edges = []
            # Hamiltonian cycle
            for i in range(N):
                edges.append((i, (i+1) % N))
            
            # Chords
            # We need chords for k=2 to 20
            for k in range(2, 21):
                edges.append((0, k))
            
            cycles = []
            for L in range(3, N+1):
                if L == N:
                    cycles.append(list(range(N)))
                else:
                    k1 = L - 1
                    k2 = N + 1 - L
                    if k1 <= 20:
                        # Use path 0->1->...->k1
                        cycles.append(list(range(0, k1+1)))
                    else:
                        # Use path 0->39->...->k2
                        # Vertices: 0, N-1, N-2, ..., k2
                        path = [0] + list(range(N-1, k2-1, -1))
                        cycles.append(path)
            
            return {'edges': edges, 'cycles': cycles}
