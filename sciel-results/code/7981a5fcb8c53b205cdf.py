import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # The optimal solution for OneMax is all ones.
    # This is a direct construction task, so we can return the optimal solution immediately.
    # We include a deadline check just to satisfy the guard requirement,
    # though it will never trigger for this O(1) operation.
    if time.time() > deadline:
        return "1" * N
    
    return "1" * N
# EVOLVE-BLOCK-END
