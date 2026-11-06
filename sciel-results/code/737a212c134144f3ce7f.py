import random
import time

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    # Safety margin: 280 seconds
    deadline = start_time + 280
    
    points = [(x, y) for x in range(N) for y in range(N)]
    best_set = []
    
    # We can run many iterations
    # Each iteration: O(N^2 * |S|^2) approx
    # With N=64, N^2=4096.
    # If |S|=20, check is 400 ops. 4096 * 400 = 1.6M ops. ~0.1s
    # If |S|=50, check is 2500 ops. 4096 * 2500 = 10M ops. ~0.5s
    # So ~200-500 iterations possible.
    
    while time.time() < deadline:
        random.shuffle(points)
        current_set = []
        
        for p in points:
            if is_compatible(p, current_set):
                current_set.append(p)
        
        if len(current_set) > len(best_set):
            best_set = current_set[:]
            
        # If we find a very good set, we might continue to see if we can improve
        # but the greedy is deterministic given the shuffle.
        
    return best_set

def is_compatible(p, S):
    """
    Returns True if p can be added to S without forming an isosceles triangle
    with any two points in S.
    """
    px, py = p
    n = len(S)
    if n < 2:
        return True
        
    for i in range(n):
        ax, ay = S[i]
        d_pa_sq = (px - ax)**2 + (py - ay)**2
        
        for j in range(n):
            if i == j:
                continue
            bx, by = S[j]
            d_pb_sq = (px - bx)**2 + (py - by)**2
            d_ab_sq = (ax - bx)**2 + (ay - by)**2
            
            # Check for isosceles conditions
            if d_pa_sq == d_pb_sq: # Apex P
                return False
            if d_pa_sq == d_ab_sq: # Apex A
                return False
            if d_pb_sq == d_ab_sq: # Apex B
                return False
                
    return True
# EVOLVE-BLOCK-END
