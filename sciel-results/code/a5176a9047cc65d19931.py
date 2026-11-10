"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def construct(q, n, radius):
    """
    Find a covering code of size as small as possible for q-ary length n radius r.
    For the specific case q=3, n=6, r=1, we aim for 73 codewords.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # The known optimal size for K_3(6,1) is 73.
    # We use a greedy construction with randomization and local search.
    
    # Strategy:
    # 1. Start with a random set of codewords (or a good heuristic start).
    # 2. Iteratively add codewords that cover the most uncovered words.
    # 3. Iteratively remove redundant codewords.
    # 4. Repeat with different random seeds to find a better solution.
    
    def hamming_distance(a, b):
        return sum(1 for i in range(n) if a[i] != b[i])
    
    def get_ball(cw):
        """Generate all words within Hamming distance 1 of cw."""
        ball = set()
        ball.add(tuple(cw))
        for i in range(n):
            for v in range(q):
                if v != cw[i]:
                    new_cw = list(cw)
                    new_cw[i] = v
                    ball.add(tuple(new_cw))
        return ball
    
    def is_covered(words_set, codewords):
        """Check if all words in words_set are covered by codewords."""
        covered = set()
        for cw in codewords:
            covered |= get_ball(cw)
        return words_set.issubset(covered)
    
    def greedy_cover():
        """
        Greedy set cover: start with empty set, add codeword that covers most uncovered words.
        """
        all_words = set(itertools.product(range(q), repeat=n))
        uncovered = set(all_words)
        codewords = []
        
        # Precompute balls for all possible codewords? No, that's 729 * 19 = 13851 words per ball.
        # Actually, let's just compute on the fly or precompute for a subset.
        
        # Better approach: iterate until all covered.
        while uncovered:
            if time.time() > deadline:
                break
            
            best_cw = None
            best_covered = 0
            
            # Try a random sample of potential codewords to speed up
            # For small spaces, we can try all.
            # 729 codewords * 19 words per ball = ~13851 operations per iteration.
            # Max 729 iterations = ~10 million ops. Should be fast enough.
            
            # To speed up, we can prioritize codewords that are in uncovered or close to uncovered.
            # But for now, let's just try all.
            
            for cw_tuple in itertools.product(range(q), repeat=n):
                ball = get_ball(list(cw_tuple))
                covered_count = len(ball & uncovered)
                if covered_count > best_covered:
                    best_covered = covered_count
                    best_cw = cw_tuple
            
            if best_cw is None:
                break
            
            codewords.append(list(best_cw))
            ball = get_ball(list(best_cw))
            uncovered -= ball
        
        return codewords
    
    def remove_redundant(codewords):
        """Remove redundant codewords."""
        if not codewords:
            return codewords
        
        all_words = set(itertools.product(range(q), repeat=n))
        
        # Precompute balls
        balls = [get_ball(cw) for cw in codewords]
        
        # Coverage count
        coverage = {}
        for i, ball in enumerate(balls):
            for w in ball:
                coverage[w] = coverage.get(w, 0) + 1
        
        removed = [False] * len(codewords)
        improved = True
        
        while improved:
            improved = False
            for i in range(len(codewords)):
                if removed[i]:
                    continue
                if time.time() > deadline:
                    return [codewords[j] for j in range(len(codewords)) if not removed[j]]
                
                can_remove = True
                for w in balls[i]:
                    if coverage[w] <= 1:
                        can_remove = False
                        break
                
                if can_remove:
                    removed[i] = True
                    improved = True
                    for w in balls[i]:
                        coverage[w] -= 1
        
        return [codewords[i] for i in range(len(codewords)) if not removed[i]]
    
    # Start with a known good construction: direct sum of tetracode
    # Tetracode: 9 words of length 4, perfect code of radius 1
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    # For n=6, we need 2 more coordinates
    # Direct sum: 9 * 3^2 = 81 codewords
    codewords_init = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    
    # Try to improve
    best_solution = remove_redundant(codewords_init)
    best_size = len(best_solution)
    
    # Try greedy cover with different randomizations
    # We'll do a few restarts
    
    for seed in range(20):
        if time.time() > deadline:
            break
        
        random.seed(seed)
        
        # Start with a random subset of the initial codewords or all of them
        # Then do greedy removal
        # Or start with empty and greedy add
        
        # Let's try: start with the 81 codewords, shuffle, and do greedy removal
        shuffled = codewords_init[:]
        random.shuffle(shuffled)
        
        solution = remove_redundant(shuffled)
        
        if len(solution) < best_size:
            best_size = len(solution)
            best_solution = solution
    
    # Try a more aggressive local search
    # We can try swapping codewords
    
    if time.time() < deadline - 30:
        # Local search: try replacing one codeword with another
        for iteration in range(50):
            if time.time() > deadline - 10:
                break
            
            improved = False
            for i in range(len(best_solution)):
                if time.time() > deadline - 10:
                    break
                
                # Try removing codeword i
                remaining = best_solution[:i] + best_solution[i+1:]
                
                # Check if still covered
                all_words = set(itertools.product(range(q), repeat=n))
                covered = set()
                for cw in remaining:
                    covered |= get_ball(cw)
                
                if all_words.issubset(covered):
                    best_solution = remaining
                    improved = True
                    break
        
            if not improved:
                break
    
    return best_solution
# EVOLVE-BLOCK-END
