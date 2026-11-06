"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 9
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110.0  # Safety margin: 10s before 120s limit
    
    def is_valid_extension(path, v):
        """Check if adding v to path keeps it an induced path."""
        if v in used:
            return False
        # v must differ from all non-consecutive vertices by at least 2 bits
        # Consecutive: path[-1] differs by 1 bit (guaranteed by construction)
        # Non-consecutive: path[0] to path[-2] must differ by >= 2 bits
        for p in path[:-1]:
            if (v ^ p).bit_count() == 1:
                return False
        return True
    
    def greedy_extend(path, used, bit_order):
        """Extend path greedily using given bit order."""
        while time.time() < deadline:
            cur = path[-1]
            nxt = None
            for b in bit_order:
                v = cur ^ (1 << b)
                if is_valid_extension(path, v):
                    nxt = v
                    break
            if nxt is None:
                break
            path.append(nxt)
            used.add(nxt)
        return path
    
    best_path = []
    best_len = 0
    
    # Try multiple random bit orderings
    for _ in range(5000):
        if time.time() >= deadline:
            break
        
        # Random bit ordering
        bit_order = list(range(N))
        random.shuffle(bit_order)
        
        path = [0]
        used = {0}
        path = greedy_extend(path, used, bit_order)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
    
    # Also try with different starting vertices
    for start in range(0, 512):  # Try several starting points
        if time.time() >= deadline:
            break
        
        for _ in range(500):
            if time.time() >= deadline:
                break
            
            bit_order = list(range(N))
            random.shuffle(bit_order)
            
            path = [start]
            used = {start}
            path = greedy_extend(path, used, bit_order)
            
            if len(path) > best_len:
                best_len = len(path)
                best_path = path[:]
    
    # Final fallback: ensure we return at least the greedy solution
    if not best_path:
        path = [0]
        used = {0}
        bit_order = list(range(N))
        best_path = greedy_extend(path, used, bit_order)
    
    return best_path
# EVOLVE-BLOCK-END
