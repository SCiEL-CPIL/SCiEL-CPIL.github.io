import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s to be safe

    best_seq = []

    def get_new_sums(x, values, sums):
        new_sums = set()
        total = x
        if total in sums:
            return None
        new_sums.add(total)
        for a in reversed(values):
            total += a
            if total in sums or total in new_sums:
                return None
            new_sums.add(total)
        return new_sums

    def is_valid(x, values, sums):
        total = x
        if total in sums:
            return False
        for a in reversed(values):
            total += a
            if total in sums:
                return False
        return True

    # 1. Beam Search
    beam = [(0, [], set())]
    K = 1000  # Beam width
    
    while beam:
        if time.time() > deadline:
            break
            
        next_beam = []
        
        for last_val, values, sums in beam:
            # Try all possible next values
            for x in range(last_val + 1, 257):
                new_sums = get_new_sums(x, values, sums)
                if new_sums is not None:
                    new_values = values + [x]
                    new_sums_set = sums | new_sums
                    # Heuristic: prefer smaller last_val (more room)
                    score = 256 - last_val
                    next_beam.append((score, last_val, new_values, new_sums_set))
            
        if not next_beam:
            break
            
        # Sort by score descending (smallest last_val first)
        next_beam.sort(key=lambda item: item[0], reverse=True)
        
        # Keep top K
        beam = []
        for item in next_beam[:K]:
            _, last_val, values, sums = item
            beam.append((last_val, values, sums))
            
            # Update best
            if len(values) > len(best_seq):
                best_seq = values

    # 2. Randomized Greedy with remaining time
    if time.time() < deadline - 2:
        seed = 42
        while time.time() < deadline - 2:
            random.seed(seed)
            values = []
            sums = set()
            while True:
                candidates = []
                last = values[-1] if values else 0
                for x in range(last + 1, 257):
                    if is_valid(x, values, sums):
                        candidates.append(x)
                if not candidates:
                    break
                # Pick a random candidate, biased towards smaller
                # Smaller candidates are better for packing
                # We can pick the smallest with prob 0.5, else random
                if random.random() < 0.5:
                    x = candidates[0]
                else:
                    x = random.choice(candidates)
                
                values.append(x)
                total = x
                sums.add(total)
                for a in reversed(values[:-1]):
                    total += a
                    sums.add(total)
            
            if len(values) > len(best_seq):
                best_seq = values
            seed += 1

    return best_seq
# EVOLVE-BLOCK-END
