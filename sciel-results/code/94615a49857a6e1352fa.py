import random

def construct():
    best_set = []
    best_size = 0
    
    def can_add(x, current_set, counts):
        s = 2 * x
        if counts[s] >= 2:
            return False
        for y in current_set:
            s = x + y
            if counts[s] >= 2:
                return False
        return True

    def add_element(x, current_set, counts):
        s = 2 * x
        counts[s] += 1
        for y in current_set:
            s = x + y
            counts[s] += 1
        current_set.append(x)

    def remove_element(x, current_set, counts):
        s = 2 * x
        counts[s] -= 1
        for y in current_set:
            if y != x:
                s = x + y
                counts[s] -= 1
        current_set.remove(x)

    def greedy_construct(order):
        current_set = []
        counts = [0] * 2001
        in_set = [False] * 1001
        for x in order:
            if not in_set[x]:
                if can_add(x, current_set, counts):
                    add_element(x, current_set, counts)
                    in_set[x] = True
        return current_set, counts, in_set

    def local_search(current_set, counts, in_set, iterations=2000):
        T = 1.0
        cooling = 0.999
        best_local_size = len(current_set)
        best_local_set = list(current_set)
        best_local_counts = counts[:]
        best_local_in_set = in_set[:]
        
        # Pre-compute list of unused? No, too expensive to update.
        # Just iterate 1..1000 to find unused?
        
        for _ in range(iterations):
            T *= cooling
            if T < 0.01:
                break
            
            r = random.random()
            if r < 0.7:
                # Try Add
                # Find a random unused
                # We can pick random x from 1..1000 and check if not in_set[x]
                # This is O(1) average if we just pick and check.
                # But if many are used, we might miss.
                # Better: maintain a list of unused indices?
                # Let's just pick random x.
                for _attempt in range(10): # Try 10 times to find an unused
                    x = random.randint(1, 1000)
                    if not in_set[x]:
                        break
                else:
                    continue
                
                if can_add(x, current_set, counts):
                    add_element(x, current_set, counts)
                    in_set[x] = True
                    if len(current_set) > best_local_size:
                        best_local_size = len(current_set)
                        best_local_set = list(current_set)
                        best_local_counts = counts[:]
                        best_local_in_set = in_set[:]
                        
            elif r < 0.95:
                # Try Swap
                if not current_set:
                    continue
                # Pick x in set
                x = random.choice(current_set)
                # Pick y not in set
                for _attempt in range(10):
                    y = random.randint(1, 1000)
                    if not in_set[y]:
                        break
                else:
                    continue
                    
                # Remove x
                remove_element(x, current_set, counts)
                in_set[x] = False
                
                if can_add(y, current_set, counts):
                    add_element(y, current_set, counts)
                    in_set[y] = True
                    # Accept neutral
                    if random.random() < 0.5:
                        pass
                    else:
                        # Revert
                        remove_element(y, current_set, counts)
                        in_set[y] = False
                        add_element(x, current_set, counts)
                        in_set[x] = True
                else:
                    # Revert
                    add_element(x, current_set, counts)
                    in_set[x] = True
            else:
                # Try Remove
                if not current_set:
                    continue
                x = random.choice(current_set)
                if random.random() < 0.05:
                    remove_element(x, current_set, counts)
                    in_set[x] = False

        return best_local_set, best_local_counts, best_local_in_set

    orders = [list(range(1, 1001)), list(range(1000, 0, -1))]
    for _ in range(5):
        o = list(range(1, 1001))
        random.shuffle(o)
        orders.append(o)

    for order in orders:
        current_set, counts, in_set = greedy_construct(order)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
            
        sa_set, sa_counts, sa_in_set = local_search(list(current_set), counts[:], in_set[:], iterations=3000)
        
        if len(sa_set) > best_size:
            best_size = len(sa_set)
            best_set = sa_set

    return best_set
