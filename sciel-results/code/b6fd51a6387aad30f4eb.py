"""Search-based snake construction for Q_9."""
import time
import random
import numpy as np

N = 9
MAX_VERTS = 2 ** N

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145  # 5s safety margin
    best_path = None
    best_len = 0
    
    def is_valid_extension(path, v, last_idx):
        """Check if v can be appended to path (v must be at distance 1 from path[-1]
        and distance >= 2 from all other vertices in path)."""
        if v in path_set:
            return False
        # v must differ from last by exactly 1 bit
        if (v ^ path[last_idx]).bit_count() != 1:
            return False
        # v must differ from all other path vertices by >= 2 bits
        for i in range(last_idx - 1):
            if (v ^ path[i]).bit_count() == 1:
                return False
        return True
    
    def greedy_extend(path, used_set, bit_order, deadline):
        """Extend path greedily using given bit order preference."""
        path = list(path)
        used_set = set(used_set)
        while time.time() < deadline:
            cur = path[-1]
            found = False
            # Try bits in the given order
            for b in bit_order:
                v = cur ^ (1 << b)
                if v in used_set:
                    continue
                # Check if v is at distance 1 from any non-adjacent vertex
                valid = True
                for i in range(len(path) - 1):
                    if (v ^ path[i]).bit_count() == 1:
                        valid = False
                        break
                if valid:
                    path.append(v)
                    used_set.add(v)
                    found = True
                    break
            if not found:
                break
        return path, used_set
    
    def randomized_greedy(seed_val, deadline):
        """Generate a snake using randomized greedy with random bit ordering."""
        rng = random.Random(seed_val)
        bit_order = list(range(N))
        rng.shuffle(bit_order)
        
        start = rng.randint(0, MAX_VERTS - 1)
        path = [start]
        used_set = {start}
        
        # Greedy extend
        while time.time() < deadline:
            cur = path[-1]
            found = False
            # Shuffle bit order slightly each step for diversity
            local_order = bit_order[:]
            rng.shuffle(local_order)
            for b in local_order:
                v = cur ^ (1 << b)
                if v in used_set:
                    continue
                # Check distance to all non-adjacent vertices
                valid = True
                for i in range(len(path) - 1):
                    if (v ^ path[i]).bit_count() == 1:
                        valid = False
                        break
                if valid:
                    path.append(v)
                    used_set.add(v)
                    found = True
                    break
            if not found:
                break
        return path
    
    def try_extend_both_ends(path, used_set, deadline):
        """Try to extend the snake from both ends."""
        path = list(path)
        used_set = set(used_set)
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Try extending from the end
            cur = path[-1]
            for b in range(N):
                v = cur ^ (1 << b)
                if v in used_set:
                    continue
                valid = True
                for i in range(len(path) - 1):
                    if (v ^ path[i]).bit_count() == 1:
                        valid = False
                        break
                if valid:
                    path.append(v)
                    used_set.add(v)
                    changed = True
                    break
            
            # Try extending from the beginning
            if not changed:
                cur = path[0]
                for b in range(N):
                    v = cur ^ (1 << b)
                    if v in used_set:
                        continue
                    valid = True
                    for i in range(1, len(path)):
                        if (v ^ path[i]).bit_count() == 1:
                            valid = False
                            break
                    if valid:
                        path.insert(0, v)
                        used_set.add(v)
                        changed = True
                        break
        
        return path
    
    # First, get a baseline greedy snake
    baseline_path = randomized_greedy(42, deadline)
    best_path = baseline_path
    best_len = len(baseline_path) - 1  # number of edges
    
    # Try many randomized restarts
    seed = 0
    while time.time() < deadline:
        seed += 1
        path = randomized_greedy(seed, deadline)
        
        # Try to extend from both ends
        used_set = set(path)
        extended = try_extend_both_ends(path, used_set, deadline)
        
        if len(extended) - 1 > best_len:
            best_len = len(extended) - 1
            best_path = extended
        
        # If we have time, try more extensions
        if time.time() < deadline - 5:
            used_set = set(best_path)
            further_extended = try_extend_both_ends(best_path, used_set, deadline)
            if len(further_extended) - 1 > best_len:
                best_len = len(further_extended) - 1
                best_path = further_extended
    
    return best_path
# EVOLVE-BLOCK-END
