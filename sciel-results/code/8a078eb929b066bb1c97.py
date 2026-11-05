import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    total_words = q ** n
    
    def hamming_dist(a, b):
        return sum(1 for i in range(n) if a[i] != b[i])
    
    def covers(codeword, word):
        return hamming_dist(codeword, word) <= radius
    
    def get_ball(codeword):
        """Get all words within radius of codeword"""
        ball = [codeword]
        for i in range(n):
            for val in range(q):
                if val == codeword[i]:
                    continue
                new_word = list(codeword)
                new_word[i] = val
                ball.append(new_word)
        return ball
    
    def is_covered(word, codewords):
        for cw in codewords:
            if hamming_dist(word, cw) <= radius:
                return True
        return False
    
    def coverage_count(codeword, codewords):
        """Count how many words are newly covered by this codeword"""
        count = 0
        for word in get_ball(codeword):
            if not is_covered(word, codewords):
                count += 1
        return count
    
    def greedy_cover():
        """Greedy set cover"""
        codewords = []
        covered = set()
        all_words = list(itertools.product(range(q), repeat=n))
        
        while len(covered) < total_words:
            best_cw = None
            best_count = -1
            for word in all_words:
                if word in covered:
                    continue
                count = 0
                for w2 in get_ball(list(word)):
                    if w2 not in covered:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_cw = word
                if time.time() > deadline:
                    break
            if best_cw is None or time.time() > deadline:
                break
            codewords.append(list(best_cw))
            for w in get_ball(list(best_cw)):
                covered.add(tuple(w))
        
        return codewords
    
    def local_search(codewords, max_iter=5000):
        """Try to improve by removing or swapping codewords"""
        codewords = [list(cw) for cw in codewords]
        best = codewords[:]
        best_size = len(best)
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            # Try removing each codeword
            for i in range(len(codewords)):
                test_cw = codewords[i]
                remaining = [cw for j, cw in enumerate(codewords) if j != i]
                
                # Check if all words are still covered
                all_covered = True
                for word in itertools.product(range(q), repeat=n):
                    if not any(hamming_dist(word, cw) <= radius for cw in remaining):
                        all_covered = False
                        break
                
                if all_covered:
                    codewords = remaining
                    improved = True
                    if len(codewords) < best_size:
                        best = codewords[:]
                        best_size = len(best)
                    break
            
            if not improved:
                # Try swapping: remove one, add another
                for i in range(len(codewords)):
                    test_cw = codewords[i]
                    remaining = [cw for j, cw in enumerate(codewords) if j != i]
                    
                    # Find a word that's poorly covered and try to add it
                    # Simple heuristic: add a random uncovered-ish word
                    for _ in range(5):
                        if time.time() > deadline:
                            break
                        new_cw = list(random.choices(range(q), k=n))
                        if new_cw in codewords:
                            continue
                        
                        test_set = remaining + [new_cw]
                        # Check coverage
                        all_covered = True
                        for word in itertools.product(range(q), repeat=n):
                            if not any(hamming_dist(word, cw) <= radius for cw in test_set):
                                all_covered = False
                                break
                        
                        if all_covered:
                            codewords = test_set
                            improved = True
                            if len(codewords) < best_size:
                                best = codewords[:]
                                best_size = len(best)
                            break
                    
                    if improved:
                        break
            
            if not improved and len(codewords) == best_size:
                # Stuck, try a random perturbation
                if random.random() < 0.1:
                    idx = random.randint(0, len(codewords) - 1)
                    pos = random.randint(0, n - 1)
                    old_val = codewords[idx][pos]
                    new_val = random.choice([v for v in range(q) if v != old_val])
                    codewords[idx][pos] = new_val
        
        return best
    
    # Initial greedy solution
    codewords = greedy_cover()
    
    # Local search improvement
    codewords = local_search(codewords)
    
    # Try a few restarts with different random seeds
    best_solution = codewords
    best_size = len(best_solution)
    
    for restart in range(10):
        if time.time() > deadline:
            break
        
        random.seed(restart * 12345 + 67890)
        # Perturb the solution
        perturbed = [list(cw) for cw in best_solution]
        if perturbed:
            idx = random.randint(0, len(perturbed) - 1)
            pos = random.randint(0, n - 1)
            old_val = perturbed[idx][pos]
            new_val = random.choice([v for v in range(q) if v != old_val])
            perturbed[idx][pos] = new_val
        
        improved = local_search(perturbed, max_iter=2000)
        
        if len(improved) < best_size:
            best_solution = improved
            best_size = len(improved)
    
    return best_solution
