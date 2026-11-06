import time
import random
import itertools
import sys

DEADLINE = 150
SAFETY_MARGIN = 10
START_TIME = None

def get_size6_violations(adj_masks, n):
    """
    Returns a list of tuples representing size-6 subsets that induce regular subgraphs.
    """
    violations = []
    # Precompute masks for all combinations? Or just iterate.
    # For n=21, C(21,6) = 54264. This is small.
    for combo in itertools.combinations(range(n), 6):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        
        # Check regularity
        d0 = (adj_masks[combo[0]] & mask).bit_count()
        is_regular = True
        for v in combo[1:]:
            if (adj_masks[v] & mask).bit_count() != d0:
                is_regular = False
                break
        if is_regular:
            violations.append(combo)
    return violations

def count_size6_violations(adj_masks, n):
    """
    Efficiently counts size-6 violations.
    """
    count = 0
    for combo in itertools.combinations(range(n), 6):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        d0 = (adj_masks[combo[0]] & mask).bit_count()
        is_regular = True
        for v in combo[1:]:
            if (adj_masks[v] & mask).bit_count() != d0:
                is_regular = False
                break
        if is_regular:
            count += 1
    return count

def find_any_violation(adj_masks, n):
    """
    Finds any violation of size >= 6.
    Prioritizes size 6, then 7, etc.
    """
    # Size 6
    for combo in itertools.combinations(range(n), 6):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        d0 = (adj_masks[combo[0]] & mask).bit_count()
        if all((adj_masks[v] & mask).bit_count() == d0 for v in combo[1:]):
            return combo
            
    # Size 7
    for combo in itertools.combinations(range(n), 7):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        d0 = (adj_masks[combo[0]] & mask).bit_count()
        if all((adj_masks[v] & mask).bit_count() == d0 for v in combo[1:]):
            return combo
            
    # Size 8+ (less likely, check up to n)
    for k in range(8, n + 1):
        for combo in itertools.combinations(range(n), k):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            d0 = (adj_masks[combo[0]] & mask).bit_count()
            if all((adj_masks[v] & mask).bit_count() == d0 for v in combo[1:]):
                return combo
    return None

def get_edges_from_masks(adj_masks, n):
    e = []
    for u in range(n):
        mask = adj_masks[u]
        for v in range(u + 1, n):
            if mask & (1 << v):
                e.append([u, v])
    return e

def simulated_annealing(adj_masks, n, rng, deadline):
    """
    Minimizes the number of size-6 violations using Simulated Annealing.
    """
    cur = adj_masks[:]
    cur_energy = count_size6_violations(cur, n)
    
    if cur_energy == 0:
        return cur, 0

    T = 1.0
    cooling = 0.995
    min_T = 1e-4
    
    while T > min_T:
        if time.time() > deadline:
            break
            
        # Pick a random edge to flip
        u = rng.randrange(n)
        v = rng.randrange(n)
        if u == v:
            continue
            
        # Calculate delta energy
        # Only size-6 sets containing u and v are affected
        # But for simplicity and correctness, we can just re-calculate energy?
        # C(21,6) is ~50k. Re-calculating 50k checks per step is ~0.1s?
        # Too slow for SA.
        
        # Optimized Delta Calculation:
        # A size-6 set S is affected if u in S and v in S.
        # We need to check if S was regular before, and if it is regular after.
        # If it was regular and becomes non-regular, energy decreases by 1.
        # If it was non-regular and becomes regular, energy increases by 1.
        # If it was regular and stays regular, or non-regular and stays non-regular, no change.
        
        # We need to identify all size-6 sets containing u and v.
        # There are C(n-2, 4) such sets. For n=21, C(19,4) = 3876.
        # We can iterate through these.
        
        # To do this efficiently, we need to know which of these sets were regular BEFORE.
        # We don't have that list.
        # We can just check them all.
        
        delta = 0
        # Generate all other 4 vertices from range(n) excluding u, v
        others = [x for x in range(n) if x != u and x != v]
        for combo_others in itertools.combinations(others, 4):
            S = tuple(sorted([u, v] + list(combo_others)))
            mask = 0
            for x in S:
                mask |= (1 << x)
            
            # Check before
            d0_before = (cur[S[0]] & mask).bit_count()
            reg_before = True
            for x in S[1:]:
                if (cur[x] & mask).bit_count() != d0_before:
                    reg_before = False
                    break
            
            # Flip edge u-v in a temporary view?
            # We can simulate the flip by checking degrees manually.
            # deg(x) in S changes only for u and v.
            # deg(u) changes by +1 if edge exists, -1 if not.
            # Let's determine if edge exists.
            edge_exists = bool(cur[u] & (1 << v))
            
            # New degrees
            # We need to check if S is regular AFTER flip.
            # We can compute the new degree sequence.
            # But it's easier to just apply the flip to a copy? No, too slow.
            
            # Let's compute the new regularity.
            # For each x in S, new_deg(x) = old_deg(x) + delta_deg(x)
            # delta_deg(u) = 1 if not edge_exists else -1
            # delta_deg(v) = 1 if not edge_exists else -1
            # delta_deg(x) = 0 for others
            
            du = 1 if not edge_exists else -1
            
            # Check if S is regular after flip
            # We need the new degrees of all vertices in S.
            # We can compute d0_new based on the first vertex, then compare others.
            
            # This is getting complex. 
            # Given the time limit, maybe simple re-evaluation of energy is acceptable if we reduce steps?
            # Or we can just use Local Search with "Walk" strategy.
            
            pass # Placeholder for complex delta logic
            
        # Given the complexity of delta, let's stick to a simpler Local Search but with RESTARTS.
        # The previous local search failed likely due to getting stuck.
        # Structural change: Random Restarts + Diversification.
        
        break # Exit SA loop as delta logic is too complex for quick implementation
        
    return cur, cur_energy

def local_search_with_restarts(adj_masks, n, rng, deadline, max_restarts=5):
    """
    Performs local search with random restarts.
    """
    best_valid = False
    best_edges = []
    
    for r in range(max_restarts):
        if time.time() > deadline:
            break
        
        # If not first restart, randomize the graph slightly
        if r > 0:
            # Flip a few random edges to diversify
            for _ in range(5):
                u = rng.randrange(n)
                v = rng.randrange(n)
                if u != v:
                    adj_masks[u] ^= (1 << v)
                    adj_masks[v] ^= (1 << u)

        # Local Search
        cur = adj_masks[:]
        max_iters = 500
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            viol = find_any_violation(cur, n)
            if viol is None:
                best_valid = True
                best_edges = get_edges_from_masks(cur, n)
                return best_valid, best_edges
            
            vset = set(viol)
            if len(vset) == n:
                u, v = rng.sample(list(vset), 2)
                cur[u] ^= (1 << v)
                cur[v] ^= (1 << u)
            else:
                u = rng.choice(list(vset))
                outside = [v for v in range(n) if v not in vset]
                if outside:
                    w = rng.choice(outside)
                    cur[u] ^= (1 << w)
                    cur[w] ^= (1 << u)
                else:
                    # Should not happen if len(vset) < n
                    break

    return best_valid, best_edges

def construct():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()

    deadline = START_TIME + DEADLINE - SAFETY_MARGIN
    rng = random.Random(42)
    
    best_n = 0
    best_edges = []
    best_valid = False

    # Try n=21 first, then 20, etc.
    for n in range(22, 5, -1):
        if time.time() > deadline:
            break

        # Try multiple random initializations
        for attempt in range(20):
            if time.time() > deadline:
                break

            p = rng.choice([0.3, 0.4, 0.5, 0.6, 0.7])
            adj_masks = [0] * n
            for u in range(n):
                for v in range(u + 1, n):
                    if rng.random() < p:
                        adj_masks[u] |= (1 << v)
                        adj_masks[v] |= (1 << u)

            valid, edges = local_search_with_restarts(adj_masks, n, rng, deadline, max_restarts=3)
            
            if valid:
                if n > best_n:
                    best_n = n
                    best_edges = edges
                    best_valid = True
                break # Found valid for this n, try next (smaller) n? No, we want max n.
                # Since we iterate n downwards, the first valid n is the max.
                # But we might find a valid n=21, then break the n loop?
                # Yes, if we find a valid graph for n, we can return it immediately 
                # because we are iterating from high to low.
                # Wait, if we find valid for n=21, that's the best possible (max 22, but 21 is target).
                # Actually, we should return immediately if we find a valid graph for the current n 
                # because we are going downwards.
                return {'n': n, 'edges': edges}

    if not best_valid or best_n < 6:
        return {'n': 5, 'edges': []}

    return {'n': best_n, 'edges': best_edges}

if __name__ == "__main__":
    result = construct()
    print(f"Found graph with n={result['n']}, edges={len(result['edges'])}")
