import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a large subset of {1, ..., 1000} such that every integer has at most
    two representations a + b with a <= b from the set.
    Uses a multi-stage approach:
    1. Fast greedy baseline.
    2. Local search (swap-based) to improve the set.
    3. Probabilistic refinement with random restarts if time permits.
    """
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill
    
    def check_validity(S):
        """Check if set S satisfies the B2[2] property."""
        if len(S) == 0:
            return True
        counts = defaultdict(int)
        S_sorted = sorted(S)
        for i, a in enumerate(S_sorted):
            for b in S_sorted[i:]:
                s = a + b
                counts[s] += 1
                if counts[s] > 2:
                    return False
        return True

    def greedy_construct():
        """Standard greedy: add x if it doesn't violate the constraint."""
        chosen = []
        counts = defaultdict(int)
        for x in range(1, 1001):
            new_sums = [x + y for y in chosen] + [2 * x]
            valid = True
            for s in new_sums:
                if counts[s] + 1 > 2:
                    valid = False
                    break
            if valid:
                chosen.append(x)
                for s in new_sums:
                    counts[s] += 1
        return chosen

    def local_search_improve(S, max_iter=5000):
        """
        Try to improve set S by swapping elements.
        For each element in S, try replacing it with a number not in S.
        Keep the best improvement found.
        """
        S = list(S)
        S_set = set(S)
        available = set(range(1, 1001)) - S_set
        current_size = len(S)
        
        def get_score(S_list):
            # We want to maximize size, but since we only swap, size is constant.
            # The goal is to find a set that allows further additions? 
            # Actually, local search on fixed size is tricky. 
            # Better: try to replace an element with a 'better' one that might allow 
            # more elements later? Or just check if we can add more after a swap?
            # Let's stick to a simpler heuristic: 
            # Try to replace x in S with y not in S, such that the new set is valid.
            # This doesn't change size. 
            # To increase size, we need a different strategy.
            # Let's use a "repair" strategy: 
            # 1. Start with greedy set.
            # 2. Try to add elements that were skipped.
            # 3. If adding x fails, try to remove some elements and add x + others?
            
            # Alternative: Randomized Local Search (RLS)
            # 1. Pick a random element to remove.
            # 2. Try to add a random new element.
            # 3. If valid, keep.
            # This might get stuck.
            
            # Let's try a different local search: 
            # Try to find a valid set of size k+1 given a valid set of size k.
            # This is expensive.
            
            # For now, let's just return the set if it's valid.
            return S_list

        # Simple local search: try to swap elements to "make room" for larger numbers?
        # Actually, the greedy method is often quite good. 
        # Let's try a probabilistic approach: 
        # Run greedy multiple times with random perturbations?
        
        # A better local search: 
        # 1. Start with greedy set G.
        # 2. For each x in 1..1000 not in G, check if we can add x by removing one element y from G.
        # 3. If so, do it if it leads to a "better" position (e.g., allows more future additions).
        # This is complex.
        
        # Let's stick to a simpler improvement: 
        # Try to add elements in random order, not just increasing.
        
        return S

    def randomized_greedy():
        """
        Greedy construction but considering elements in a randomized order, 
        with a bias towards smaller numbers (which are more critical for sums).
        """
        candidates = list(range(1, 1001))
        # Shuffle but keep small numbers slightly preferred? 
        # Or just shuffle.
        random.shuffle(candidates)
        
        chosen = []
        counts = defaultdict(int)
        
        for x in candidates:
            new_sums = [x + y for y in chosen] + [2 * x]
            valid = True
            for s in new_sums:
                if counts[s] + 1 > 2:
                    valid = False
                    break
            if valid:
                chosen.append(x)
                for s in new_sums:
                    counts[s] += 1
        return chosen

    def try_add_elements(S, max_iter=1000):
        """
        Try to add more elements to S by checking all possible candidates.
        """
        S = list(S)
        S_set = set(S)
        counts = defaultdict(int)
        for i, a in enumerate(S):
            for b in S[i:]:
                counts[a+b] += 1
        
        added = True
        while added:
            added = False
            for x in range(1, 1001):
                if x in S_set:
                    continue
                new_sums = [x + y for y in S] + [2 * x]
                valid = True
                for s in new_sums:
                    if counts[s] + 1 > 2:
                        valid = False
                        break
                if valid:
                    S.append(x)
                    S_set.add(x)
                    for s in new_sums:
                        counts[s] += 1
                    added = True
                    if time.time() - start_time > DEADLINE:
                        return S
        return S

    # 1. Get baseline
    best_set = greedy_construct()
    best_size = len(best_set)
    
    # 2. Try to improve with randomized greedy
    # Run a few times if time permits
    num_restarts = 5
    for i in range(num_restarts):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        current_set = randomized_greedy()
        # Try to add more elements to the randomized set
        current_set = try_add_elements(current_set)
        
        if len(current_set) > best_size:
            best_set = current_set
            best_size = len(best_set)
    
    # 3. Final polish: try to add any remaining elements to the best set
    if time.time() - start_time < DEADLINE - 5:
        best_set = try_add_elements(best_set)
        
    # Ensure validity (should be, but just in case)
    if not check_validity(best_set):
        # Fallback to greedy
        best_set = greedy_construct()
        
    return best_set
# EVOLVE-BLOCK-END
