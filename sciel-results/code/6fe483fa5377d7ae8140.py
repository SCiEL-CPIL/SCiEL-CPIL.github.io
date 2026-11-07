import random
import time
from itertools import combinations

# EVOLVE-BLOCK-START

P = {'n': 128}
MAX_TIME = 85.0

def construct():
    start_time = time.time()
    best_set = set()
    n = P['n']
    
    # Precompute pairs and their sums for faster lookup?
    # Actually, direct check is fine for small N.
    
    def is_valid(S):
        """Check if set S satisfies the condition."""
        S_list = list(S)
        for a, b, c in combinations(S_list, 3):
            if (a + b) in S and (a + c) in S and (b + c) in S:
                return False
        return True

    def can_add(S, x):
        """Check if adding x to S keeps it valid.
        Only need to check triples involving x.
        Triple {x, a, b} with a, b in S.
        Sums: x+a, x+b, a+b.
        Conflict if (x+a in S or x+a==x) and (x+b in S or x+b==x) and (a+b in S or a+b==x).
        Since a,b >= 1, x+a != x and x+b != x.
        So conflict if (x+a in S) and (x+b in S) and (a+b in S).
        """
        for a in S:
            for b in S:
                if a == b:
                    continue
                if (x + a) in S and (x + b) in S and (a + b) in S:
                    return False
        return True

    def greedy_construct(random_seed):
        """Build a set using greedy addition."""
        random.seed(random_seed)
        S = set()
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        # Heuristic: Try to add elements. 
        # To maximize size, we should probably prefer elements that cause fewer conflicts.
        # But simple random shuffle + greedy is a good baseline for diversity.
        
        # Alternative: Sort by value? The seed used high values.
        # Let's try random order.
        
        for x in candidates:
            if can_add(S, x):
                S.add(x)
        
        return S

    def local_search(S, time_limit):
        """Improve set S using local search (additions and kicks)."""
        S = set(S)
        no_improve_count = 0
        last_improve_time = time.time()
        
        while time.time() - start_time < time_limit:
            improved = False
            
            # 1. Try to add any element
            # To speed up, we can shuffle candidates to avoid bias
            candidates = [x for x in range(1, n + 1) if x not in S]
            random.shuffle(candidates)
            
            added = False
            for x in candidates:
                if can_add(S, x):
                    S.add(x)
                    improved = True
                    added = True
                    break # Restart loop to re-evaluate candidates
            
            if added:
                no_improve_count = 0
                last_improve_time = time.time()
                continue
            
            # 2. If no addition possible, try a "kick" (remove an element)
            # Remove an element that is "most responsible" for blocking others?
            # Heuristic: Remove an element involved in the most potential conflicts.
            # Or just remove a random element to escape local optimum.
            
            if S:
                # Calculate a score for each element in S: how many other elements would become addable if it were removed?
                # This is expensive. Let's just pick a random one or one with high value.
                # High values might be less restrictive?
                
                # Let's try removing the element that participates in the most "almost conflicts"
                # For simplicity, random removal is effective with many restarts.
                # But let's try a slightly smarter one:
                # Count for each y in S, how many x not in S are blocked ONLY by y?
                # Too complex. Let's use random kick.
                
                victim = random.choice(list(S))
                S.remove(victim)
                no_improve_count += 1
                
                # If we've been stuck for too long, force a more drastic change or restart
                if no_improve_count > 50:
                    break
            else:
                break
                
        return S

    # Main Loop: Multiple restarts
    best_size = 0
    best_set_final = set()
    
    # Run several iterations with different random seeds
    # Each iteration: Greedy -> Local Search
    # We can also mix in "Start with seed solution and modify"
    
    seed_solution = set(range(n//2 + 1, n + 1))
    
    # Try improving the seed solution first
    improved_seed = local_search(seed_solution, start_time + 5.0)
    if len(improved_seed) > best_size:
        best_size = len(improved_seed)
        best_set_final = improved_seed
        
    # Try random restarts
    while time.time() - start_time < MAX_TIME:
        # Greedy construction with random order
        S = greedy_construct(random.randint(0, 1000000))
        
        # Local search to improve
        S = local_search(S, time.time() + 5.0) # 5s per restart
        
        if len(S) > best_size:
            best_size = len(S)
            best_set_final = S
            
        # Check time
        if time.time() - start_time > MAX_TIME - 5:
            break

    # Final check: ensure validity of the best set found
    # The local search should maintain validity, but let's be safe.
    if not is_valid(best_set_final):
        # Fallback to seed if something went wrong
        return list(range(n//2 + 1, n + 1))
        
    return list(best_set_final)

# EVOLVE-BLOCK-END
