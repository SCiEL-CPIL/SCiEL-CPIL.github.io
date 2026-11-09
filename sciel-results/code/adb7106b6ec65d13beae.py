"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    N = 1000000
    phi = list(range(N + 1))
    
    for i in range(2, N + 1):
        if phi[i] == i:
            for j in range(i, N + 1, i):
                phi[j] -= phi[j] // i
                
    for n in range(1, N + 1):
        if phi[n] % 120 == 0:
            return n
            
    return 14400
# EVOLVE-BLOCK-END
