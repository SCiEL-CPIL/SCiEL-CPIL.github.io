import random

N = 100

# EVOLVE-BLOCK-START
def construct():
    # Start with a random binary string
    bits = [random.randint(0, 1) for _ in range(N)]
    
    # Greedy improvement: flip any 0 to 1 to increase the count of ones
    # Since the objective is to maximize ones, flipping 0->1 is always beneficial.
    for i in range(N):
        if bits[i] == 0:
            bits[i] = 1
            
    return "".join(map(str, bits))
# EVOLVE-BLOCK-END
