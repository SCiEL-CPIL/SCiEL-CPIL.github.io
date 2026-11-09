"""Snake-in-the-box in Q_9: find the longest induced path."""
import time
import random

N = 9
NUM_VERTICES = 1 << N  # 512
DEADLINE = 140.0  # safety margin before 150s kill

def construct():
    start_time = time.time()
    best_path = None
    best_len = 0
    
    bit_masks = [1 << b for b in range(N)]
    
    def greedy(bit_order):
        """Greedy snake construction with given bit ordering."""
        path = [0]
        used = {0}
        neighbor_set = set()  # neighbors of all vertices in path except the last
        
        while True:
            cur = path[-1]
            nxt = None
            for b in bit_order:
                v = cur ^ bit_masks[b]
                if v in used or v in neighbor_set:
                    continue
                nxt = v
                break
            if nxt is None:
                break
            # The previous last vertex cur becomes non-last; forbid its neighbors
            for mb in bit_masks:
                neighbor_set.add(cur ^ mb)
            path.append(nxt)
            used.add(nxt)
        
        return path
    
    # Try many random bit orderings
    rng = random.Random(12345)
    
    while time.time() - start_time < DEADLINE:
        bit_order = list(range(N))
        rng.shuffle(bit_order)
        
        path = greedy(bit_order)
        length = len(path) - 1
        
        if length > best_len:
            best_len = length
            best_path = path
        
        if best_len >= 150:
            break
    
    if best_path is None:
        # Fallback: simple greedy with natural bit order
        best_path = greedy(list(range(N)))
    
    return best_path
