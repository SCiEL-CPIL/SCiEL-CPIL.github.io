import time
import random
import sys

sys.setrecursionlimit(10000)

def is_valid(seq):
    n = len(seq)
    if n == 0:
        return True
    products = set()
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            if prod in products:
                return False
            products.add(prod)
    return True

def get_products(seq):
    n = len(seq)
    products = set()
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            products.add(prod)
    return products

def can_insert(seq, x, pos):
    """Check if x can be inserted at position pos in seq."""
    new_seq = seq[:pos] + [x] + seq[pos:]
    if new_seq != sorted(new_seq):
        return False
    # Check all products
    n = len(new_seq)
    products = set()
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= new_seq[j]
            if prod in products:
                return False
            products.add(prod)
    return True

def construct():
    deadline = time.time() + 85.0
    best_seq = []
    best_len = 0
    
    # Strategy 1: Ascending Greedy (Seed)
    values, products = [], set()
    for x in range(2, 129):
        tails, prod = [x], x
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    if len(values) > best_len:
        best_len = len(values)
        best_seq = values[:]
    
    # Strategy 2: Descending Greedy
    candidates = list(range(128, 1, -1))
    current_set = set()
    for x in candidates:
        new_set = current_set | {x}
        new_seq = sorted(new_set)
        if is_valid(new_seq):
            current_set = new_set
    seq = sorted(current_set)
    if len(seq) > best_len:
        best_len = len(seq)
        best_seq = seq[:]
    
    # Strategy 3: Randomized greedy with multiple restarts
    def randomized_greedy(order):
        current_set = set()
        for x in order:
            new_set = current_set | {x}
            new_seq = sorted(new_set)
            if is_valid(new_seq):
                current_set = new_set
        return sorted(current_set)
    
    # Run several randomized greedy passes
    for _ in range(30):
        if time.time() > deadline:
            break
        candidates = list(range(2, 129))
        random.shuffle(candidates)
        seq = randomized_greedy(candidates)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
    
    # Strategy 4: Local search improvement on best_seq
    def local_search(seq, max_iter=2000):
        seq = list(seq)
        current_len = len(seq)
        missing = [x for x in range(2, 129) if x not in seq]
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try to insert a missing element
            if missing:
                random.shuffle(missing)
                for x in missing[:20]:  # Try a few
                    # Try all possible positions
                    for pos in range(len(seq) + 1):
                        if pos > 0 and seq[pos-1] >= x:
                            continue
                        if pos < len(seq) and seq[pos] <= x:
                            continue
                        new_seq = seq[:pos] + [x] + seq[pos:]
                        if is_valid(new_seq):
                            seq = new_seq
                            missing.remove(x)
                            current_len += 1
                            improved = True
                            break
                    if improved:
                        break
            
            # Try to remove an element to allow better insertion
            if not improved and current_len > 10:
                # Try removing a random element and see if we can add two back
                idx = random.randint(0, len(seq) - 1)
                removed = seq[idx]
                seq_without = seq[:idx] + seq[idx+1:]
                missing_with = missing + [removed]
                
                # Try to add back two elements
                added_count = 0
                for x in missing_with[:15]:
                    for pos in range(len(seq_without) + 1):
                        if pos > 0 and seq_without[pos-1] >= x:
                            continue
                        if pos < len(seq_without) and seq_without[pos] <= x:
                            continue
                        new_seq = seq_without[:pos] + [x] + seq_without[pos:]
                        if is_valid(new_seq):
                            seq_without = new_seq
                            missing_with.remove(x)
                            added_count += 1
                            break
                    if added_count >= 2:
                        break
                
                if added_count >= 2:
                    seq = seq_without
                    current_len = len(seq)
                    missing = [x for x in range(2, 129) if x not in seq]
                    improved = True
            
            if not improved and not missing:
                break
        
        return seq
    
    # Apply local search to best_seq
    if best_len < 127:
        improved_seq = local_search(best_seq, max_iter=3000)
        if len(improved_seq) > best_len:
            best_len = len(improved_seq)
            best_seq = improved_seq
    
    # Strategy 5: More randomized restarts with local search
    for _ in range(50):
        if time.time() > deadline:
            break
        candidates = list(range(2, 129))
        random.shuffle(candidates)
        seq = randomized_greedy(candidates)
        
        # Quick local search
        improved_seq = local_search(seq, max_iter=500)
        if len(improved_seq) > best_len:
            best_len = len(improved_seq)
            best_seq = improved_seq[:]
    
    return best_seq
