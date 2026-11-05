import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin before 120s kill
    
    prescribed = {1, 2, 5, 11}
    N = 284
    TARGET_SIZE = 20

    def get_sums_bitset(s):
        """Returns an integer bitset where bit i is set if sum i is present."""
        bitset = 0
        n = len(s)
        for i in range(n):
            for j in range(i, n):
                bitset |= 1 << (s[i] + s[j])
        return bitset

    def is_valid(s):
        """Check if s is a valid Sidon set."""
        sums = set()
        n = len(s)
        for i in range(n):
            for j in range(i, n):
                val = s[i] + s[j]
                if val in sums:
                    return False
                sums.add(val)
        return True

    def greedy_construct():
        """Construct a valid Sidon set greedily."""
        s = sorted(prescribed)
        sums_bitset = get_sums_bitset(s)
        
        for x in range(1, N + 1):
            if x in s:
                continue
            
            # Check if x can be added using bitset
            ok = True
            for y in s:
                if (sums_bitset >> (x + y)) & 1:
                    ok = False
                    break
            
            if ok:
                if (sums_bitset >> (2 * x)) & 1:
                    ok = False
            
            if ok:
                # Update sums_bitset
                for y in s:
                    sums_bitset |= 1 << (x + y)
                sums_bitset |= 1 << (2 * x)
                s.append(x)
                
        return s, sums_bitset

    # 1. Get a baseline valid set
    best_set, best_sums = greedy_construct()
    best_size = len(best_set)
    
    if best_size >= TARGET_SIZE:
        return best_set

    current_set = best_set[:]
    current_sums = best_sums
    
    candidates = list(range(1, N + 1))
    
    no_improve_count = 0
    kick_threshold = 5000 # Number of moves without improvement before a kick
    
    while time.time() < deadline:
        if no_improve_count > kick_threshold:
            # Kick: Remove 1 or 2 random non-prescribed elements to escape local optimum
            non_prescribed = [x for x in current_set if x not in prescribed]
            if non_prescribed:
                k = random.randint(1, 2)
                to_remove = random.sample(non_prescribed, min(k, len(non_prescribed)))
                for x in to_remove:
                    current_set.remove(x)
                current_sums = get_sums_bitset(current_set)
            no_improve_count = 0
            continue

        move_type = random.random()
        
        if move_type < 0.5:
            # Try Add
            x = random.choice(candidates)
            if x in current_set:
                no_improve_count += 1
                continue
            
            # Check validity of adding x
            valid = True
            for y in current_set:
                if (current_sums >> (x + y)) & 1:
                    valid = False
                    break
            
            if valid and ((current_sums >> (2 * x)) & 1):
                valid = False
                
            if valid:
                # Apply add
                new_set = current_set + [x]
                new_sums = current_sums
                for y in current_set:
                    new_sums |= 1 << (x + y)
                new_sums |= 1 << (2 * x)
                
                current_set = new_set
                current_sums = new_sums
                no_improve_count = 0
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set[:]
                    if best_size >= TARGET_SIZE:
                        return best_set
            else:
                no_improve_count += 1

        elif move_type < 0.9:
            # Try Swap
            non_prescribed_indices = [i for i, x in enumerate(current_set) if x not in prescribed]
            if not non_prescribed_indices:
                no_improve_count += 1
                continue
                
            idx = random.choice(non_prescribed_indices)
            old_val = current_set[idx]
            
            new_val = random.choice(candidates)
            if new_val == old_val or new_val in current_set:
                no_improve_count += 1
                continue
            
            # Check validity of swap
            new_set = current_set[:]
            new_set[idx] = new_val
            
            # Full validity check for swap
            sums = set()
            valid = True
            n = len(new_set)
            for i in range(n):
                for j in range(i, n):
                    val = new_set[i] + new_set[j]
                    if val in sums:
                        valid = False
                        break
                    sums.add(val)
                if not valid:
                    break
            
            if valid:
                current_set = new_set
                current_sums = get_sums_bitset(current_set)
                no_improve_count = 0
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set[:]
                    if best_size >= TARGET_SIZE:
                        return best_set
            else:
                no_improve_count += 1

        else:
            # Try Remove (to allow different structure)
            non_prescribed_indices = [i for i, x in enumerate(current_set) if x not in prescribed]
            if non_prescribed_indices:
                idx = random.choice(non_prescribed_indices)
                current_set.pop(idx)
                current_sums = get_sums_bitset(current_set)
                no_improve_count += 1
            else:
                no_improve_count += 1

    return best_set
