"""Combined program: greedy baseline + enhanced local-search with multi-element moves."""
import time
import random

# EVOLVE-BLOCK-START

def construct():
    start = time.time()
    deadline = 85.0  # safety margin before 90s limit

    def is_valid(seq):
        """Check if all consecutive block products are distinct."""
        prods = set()
        n = len(seq)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                if p in prods:
                    return False
                prods.add(p)
        return True

    def greedy_init():
        """Standard greedy: iterate 2..128, add if no tail-product collision."""
        values = []
        products = set()
        for x in range(2, 129):
            tails = [x]
            prod = x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        return values

    def get_neighbor(seq, rng):
        """Generate a random neighbor using various move types."""
        n = len(seq)
        if n == 0:
            return seq[:]
        
        # Choose move type with weighted probabilities
        r = rng.random()
        if r < 0.25:
            op = 'insert'
        elif r < 0.40:
            op = 'delete'
        elif r < 0.55:
            op = 'swap'
        elif r < 0.75:
            op = 'one_out_two_in'
        elif r < 0.90:
            op = 'two_out_two_in'
        else:
            op = 'one_out_one_in'
        
        if op == 'insert':
            candidates = [x for x in range(2, 129) if x not in seq]
            if not candidates:
                return seq[:]
            x = rng.choice(candidates)
            pos = rng.randint(0, n)
            new_seq = seq[:pos] + [x] + seq[pos:]
            for i in range(len(new_seq) - 1):
                if new_seq[i] >= new_seq[i + 1]:
                    return seq[:]
            return new_seq
            
        elif op == 'delete':
            if n <= 1:
                return seq[:]
            pos = rng.randint(0, n - 1)
            return seq[:pos] + seq[pos + 1:]
            
        elif op == 'swap':
            if n < 2:
                return seq[:]
            i, j = rng.sample(range(n), 2)
            new_seq = seq[:]
            new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
            for k in range(len(new_seq) - 1):
                if new_seq[k] >= new_seq[k + 1]:
                    return seq[:]
            return new_seq
            
        elif op == 'one_out_two_in':
            # Remove one element, insert two new ones
            if n < 1:
                return seq[:]
            remove_pos = rng.randint(0, n - 1)
            remaining = seq[:remove_pos] + seq[remove_pos + 1:]
            removed_val = seq[remove_pos]
            candidates = [x for x in range(2, 129) if x not in remaining and x != removed_val]
            if len(candidates) < 2:
                return seq[:]
            x1, x2 = rng.sample(candidates, 2)
            if x1 > x2:
                x1, x2 = x2, x1
            pos1 = rng.randint(0, len(remaining))
            pos2 = rng.randint(pos1 + 1, len(remaining) + 1)
            new_seq = remaining[:pos1] + [x1] + remaining[pos1:pos2] + [x2] + remaining[pos2:]
            for i in range(len(new_seq) - 1):
                if new_seq[i] >= new_seq[i + 1]:
                    return seq[:]
            return new_seq
            
        elif op == 'two_out_two_in':
            # Remove two elements, insert two new ones
            if n < 2:
                return seq[:]
            i, j = rng.sample(range(n), 2)
            if i > j:
                i, j = j, i
            removed = [seq[i], seq[j]]
            remaining = seq[:i] + seq[i+1:j] + seq[j+1:]
            candidates = [x for x in range(2, 129) if x not in remaining and x not in removed]
            if len(candidates) < 2:
                return seq[:]
            x1, x2 = rng.sample(candidates, 2)
            if x1 > x2:
                x1, x2 = x2, x1
            pos1 = rng.randint(0, len(remaining))
            pos2 = rng.randint(pos1 + 1, len(remaining) + 1)
            new_seq = remaining[:pos1] + [x1] + remaining[pos1:pos2] + [x2] + remaining[pos2:]
            for i in range(len(new_seq) - 1):
                if new_seq[i] >= new_seq[i + 1]:
                    return seq[:]
            return new_seq
            
        else:  # one_out_one_in
            # Remove one element, insert one new one (replacement)
            if n < 1:
                return seq[:]
            remove_pos = rng.randint(0, n - 1)
            removed_val = seq[remove_pos]
            remaining = seq[:remove_pos] + seq[remove_pos + 1:]
            candidates = [x for x in range(2, 129) if x not in remaining and x != removed_val]
            if not candidates:
                return seq[:]
            x = rng.choice(candidates)
            pos = rng.randint(0, len(remaining))
            new_seq = remaining[:pos] + [x] + remaining[pos:]
            for i in range(len(new_seq) - 1):
                if new_seq[i] >= new_seq[i + 1]:
                    return seq[:]
            return new_seq

    # Strategy 1: Greedy baseline
    best = greedy_init()
    best_len = len(best)

    # Strategy 2: Enhanced local search with multi-element moves
    rng = random.Random(42)
    current = best[:]
    current_len = best_len
    no_improve = 0
    max_no_improve = 8000

    while time.time() < deadline - 5 and no_improve < max_no_improve:
        neighbor = get_neighbor(current, rng)
        if len(neighbor) < current_len - 1:
            # Allow shorter neighbors occasionally for exploration
            if rng.random() > 0.15:
                no_improve += 1
                continue
        if is_valid(neighbor):
            if len(neighbor) > current_len:
                current = neighbor
                current_len = len(neighbor)
                no_improve = 0
                if current_len > best_len:
                    best = current[:]
                    best_len = current_len
            else:
                # Accept same-length or slightly shorter with some probability
                if len(neighbor) == current_len and rng.random() < 0.4:
                    current = neighbor
                no_improve += 1
        else:
            no_improve += 1

    # Strategy 3: Try different random seeds for local search
    for seed in [123, 456, 789, 1011, 1013]:
        if time.time() > deadline - 10:
            break
        rng2 = random.Random(seed)
        cur = best[:]
        cur_len = len(cur)
        no_imp = 0
        max_iter = 5000
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            if no_imp > 5000:
                break
            nb = get_neighbor(cur, rng2)
            if is_valid(nb):
                if len(nb) > cur_len:
                    cur = nb
                    cur_len = len(nb)
                    no_imp = 0
                    if cur_len > best_len:
                        best = cur[:]
                        best_len = cur_len
                else:
                    no_imp += 1
            else:
                no_imp += 1

    # Strategy 4: Targeted insertion attempts on best sequence
    if time.time() < deadline - 5:
        cur = best[:]
        cur_len = len(cur)
        # Try all possible single insertions in all positions
        for x in range(2, 129):
            if time.time() > deadline - 5:
                break
            if x in cur:
                continue
            for pos in range(len(cur) + 1):
                if time.time() > deadline - 5:
                    break
                # Check if insertion maintains strictly increasing order
                if pos > 0 and cur[pos - 1] >= x:
                    continue
                if pos < len(cur) and cur[pos] <= x:
                    continue
                candidate = cur[:pos] + [x] + cur[pos:]
                if is_valid(candidate):
                    cur = candidate
                    cur_len += 1
                    if cur_len > best_len:
                        best = cur[:]
                        best_len = cur_len
                    break  # Try next x

    # Final verification
    if not is_valid(best):
        best = greedy_init()

    return best

# EVOLVE-BLOCK-END
