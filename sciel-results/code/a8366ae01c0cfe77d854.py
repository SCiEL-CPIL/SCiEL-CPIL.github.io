import time
import random
import math

def construct():
    start = time.time()
    deadline = start + 82.0  # safety margin before 90s kill
    N = 180

    # Precompute all forbidden triples: distinct a,b,c with 1/a = 1/b + 1/c
    triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            d = b - a
            if (a * b) % d == 0:
                c = (a * b) // d
                if 1 <= c <= N and c != a and c != b:
                    triples.append((a, b, c))
    triples = list(set(triples))
    num_triples = len(triples)

    # For each element, list of (other1, other2) pairs from triples containing it
    elem_triples = [[] for _ in range(N + 1)]
    for (a, b, c) in triples:
        elem_triples[a].append((b, c))
        elem_triples[b].append((a, c))
        elem_triples[c].append((a, b))

    # Element degree (number of triples containing it)
    elem_degree = [0] * (N + 1)
    for (a, b, c) in triples:
        elem_degree[a] += 1
        elem_degree[b] += 1
        elem_degree[c] += 1

    # Build a conflict graph: edge (u,v) if there exists w such that (u,v,w) is a triple
    # This helps with faster checks in some contexts
    conflict_pairs = set()
    for (a, b, c) in triples:
        conflict_pairs.add((a, b))
        conflict_pairs.add((a, c))
        conflict_pairs.add((b, c))

    def can_add(z, S_mask):
        """Check if element z can be added to S_mask without creating a triple."""
        for (u, v) in elem_triples[z]:
            if (S_mask >> u) & 1 and (S_mask >> v) & 1:
                return False
        return True

    def count_violations(S_mask):
        """Count forbidden triples fully contained in S_mask."""
        count = 0
        for (a, b, c) in triples:
            if (S_mask >> a) & 1 and (S_mask >> b) & 1 and (S_mask >> c) & 1:
                count += 1
        return count

    def is_valid(S_mask):
        return count_violations(S_mask) == 0

    def greedy_hitting_set_randomized(seed_val):
        """
        Greedy removal from full set with randomized tie-breaking.
        Returns a valid set (as bitmask).
        """
        random.seed(seed_val)
        S_mask = 0
        for i in range(1, N + 1):
            S_mask |= (1 << i)
        
        # Track active triples count per element
        # We'll iterate until no violations
        
        max_rounds = 1000
        for _ in range(max_rounds):
            if time.time() > deadline - 5:
                break
            
            # Find violated triples
            violated = []
            for (a, b, c) in triples:
                if (S_mask >> a) & 1 and (S_mask >> b) & 1 and (S_mask >> c) & 1:
                    violated.append((a, b, c))
            
            if not violated:
                break
            
            # Count how many violated triples each element is in
            elem_count = {}
            for (a, b, c) in violated:
                elem_count[a] = elem_count.get(a, 0) + 1
                elem_count[b] = elem_count.get(b, 0) + 1
                elem_count[c] = elem_count.get(c, 0) + 1
            
            # Find max count
            max_c = max(elem_count.values())
            candidates = [e for e, cnt in elem_count.items() if cnt == max_c]
            
            # Randomly pick one to remove
            to_remove = random.choice(candidates)
            S_mask &= ~(1 << to_remove)
        
        # Polish: try to add back any removed element
        for z in range(1, N + 1):
            if not (S_mask >> z) & 1:
                if can_add(z, S_mask):
                    S_mask |= (1 << z)
        
        return S_mask

    def greedy_add_order(order_list):
        """
        Greedy add elements in a specific order.
        """
        S_mask = 0
        for z in order_list:
            if can_add(z, S_mask):
                S_mask |= (1 << z)
        return S_mask

    def local_search_polish(S_mask, max_iters=5000):
        """
        Local search: try single swaps (remove x, add y) or double swaps to increase size.
        Focuses on improving the size of a valid set.
        """
        if time.time() > deadline - 5:
            return S_mask
        
        current_size = bin(S_mask).count('1')
        best_S = S_mask
        best_size = current_size
        
        # Precompute list of elements in S and not in S for faster access
        # We will update these incrementally or recompute if needed, but for simplicity in SA,
        # we might just use bit operations.
        
        # Simple hill climbing / local search
        # Try to add any element first
        changed = True
        while changed:
            changed = False
            for z in range(1, N + 1):
                if not (S_mask >> z) & 1:
                    if can_add(z, S_mask):
                        S_mask |= (1 << z)
                        current_size += 1
                        changed = True
        
        if current_size > best_size:
            best_S = S_mask
            best_size = current_size
        
        # Now try swaps: remove one, add one
        # This is expensive if done naively. 
        # Let's try a limited number of random swaps.
        in_S = [z for z in range(1, N + 1) if (S_mask >> z) & 1]
        not_in_S = [z for z in range(1, N + 1) if not (S_mask >> z) & 1]
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            if not in_S or not not_in_S:
                break
            
            x = random.choice(in_S)
            y = random.choice(not_in_S)
            
            # Check if removing x and adding y is valid and results in a valid set
            # Actually, we want to see if we can add y. But we already tried adding.
            # So we need to remove x to free up space for y?
            # If y cannot be added to S, it means there is a triple (y, u, v) with u,v in S.
            # If we remove one of u or v, maybe y can be added.
            
            # Let's check if y can be added if we remove x.
            # This is only useful if x is one of the blockers for y.
            
            # Check blockers for y
            blockers = []
            for (u, v) in elem_triples[y]:
                if (S_mask >> u) & 1 and (S_mask >> v) & 1:
                    blockers.append(u)
                    blockers.append(v)
            
            if not blockers:
                # Should have been added already
                continue
            
            # If x is in blockers, try removing x and adding y
            if x in blockers:
                # Remove x
                new_S = S_mask & ~(1 << x)
                # Check if y can be added to new_S
                if can_add(y, new_S):
                    new_S |= (1 << y)
                    # Size is same, but maybe this allows adding MORE elements later?
                    # For now, let's just accept if it doesn't break validity (it won't) 
                    # and hope it leads to a better state. 
                    # Actually, since size is same, we should only accept if it enables future growth.
                    # Let's just do a quick check: can we add any other element to new_S?
                    # This is getting complex. Let's stick to a simpler metric:
                    # Accept if it allows adding at least one more element immediately.
                    
                    # Try to add any element to new_S
                    improved = False
                    for z in range(1, N + 1):
                        if not (new_S >> z) & 1 and z != y:
                            if can_add(z, new_S):
                                improved = True
                                break
                    
                    if improved:
                        S_mask = new_S
                        in_S = [z for z in range(1, N + 1) if (S_mask >> z) & 1]
                        not_in_S = [z for z in range(1, N + 1) if not (S_mask >> z) & 1]
                        # Re-polish
                        changed2 = True
                        while changed2:
                            changed2 = False
                            for z in range(1, N + 1):
                                if not (S_mask >> z) & 1:
                                    if can_add(z, S_mask):
                                        S_mask |= (1 << z)
                                        current_size += 1
                                        changed2 = True
                        if current_size > best_size:
                            best_S = S_mask
                            best_size = current_size
        
        return best_S

    # --- Multi-start strategy ---
    best_mask = 0
    best_size = 0

    # 1. Multiple randomized greedy hitting sets
    for i in range(30):
        if time.time() > deadline - 15:
            break
        mask = greedy_hitting_set_randomized(i)
        size = bin(mask).count('1')
        if size > best_size and is_valid(mask):
            best_mask = mask
            best_size = size

    # 2. Greedy adds with different orders
    # Order by degree ascending (pick low degree first)
    order_asc = sorted(range(1, N + 1), key=lambda x: elem_degree[x])
    mask = greedy_add_order(order_asc)
    size = bin(mask).count('1')
    if size > best_size and is_valid(mask):
        best_mask = mask
        best_size = size

    # Order by degree descending
    order_desc = sorted(range(1, N + 1), key=lambda x: -elem_degree[x])
    mask = greedy_add_order(order_desc)
    size = bin(mask).count('1')
    if size > best_size and is_valid(mask):
        best_mask = mask
        best_size = size

    # Random orders
    for i in range(10):
        if time.time() > deadline - 15:
            break
        order = list(range(1, N + 1))
        random.seed(i + 100)
        random.shuffle(order)
        mask = greedy_add_order(order)
        size = bin(mask).count('1')
        if size > best_size and is_valid(mask):
            best_mask = mask
            best_size = size

    # 3. Polish the best candidate found so far with local search
    if best_size > 0 and time.time() < deadline - 10:
        polished = local_search_polish(best_mask, max_iters=10000)
        p_size = bin(polished).count('1')
        if p_size > best_size and is_valid(polished):
            best_mask = polished
            best_size = p_size

    # 4. If we still have time, try a few more local search passes from different starts
    if time.time() < deadline - 5:
        # Try local search on the greedy add results too
        for i in range(5):
            if time.time() > deadline - 2:
                break
            # Pick a random initialization
            S_mask = 0
            for z in range(1, N + 1):
                if random.random() < 0.5:
                    S_mask |= (1 << z)
            
            # Repair it
            # Simple repair: remove elements causing conflicts
            changed = True
            while changed:
                changed = False
                for (a, b, c) in triples:
                    if (S_mask >> a) & 1 and (S_mask >> b) & 1 and (S_mask >> c) & 1:
                        # Remove one of them, say the one with highest degree
                        max_deg = -1
                        max_elem = -1
                        for e in (a, b, c):
                            if elem_degree[e] > max_deg:
                                max_deg = elem_degree[e]
                                max_elem = e
                        S_mask &= ~(1 << max_elem)
                        changed = True
            
            # Polish
            polished = local_search_polish(S_mask, max_iters=2000)
            p_size = bin(polished).count('1')
            if p_size > best_size and is_valid(polished):
                best_mask = polished
                best_size = p_size

    # Final polish on the absolute best
    if best_size > 0 and time.time() < deadline - 1:
        # One last try to add any element
        for z in range(1, N + 1):
            if not (best_mask >> z) & 1:
                if can_add(z, best_mask):
                    best_mask |= (1 << z)
        final_size = bin(best_mask).count('1')
        if final_size > best_size:
            best_size = final_size

    # Convert bitmask to list
    result = []
    for z in range(1, N + 1):
        if (best_mask >> z) & 1:
            result.append(z)

    if not result:
        result = [1]

    return result
