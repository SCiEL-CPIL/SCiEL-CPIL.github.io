import random

def construct():
    # Initial set: random or interval
    # Interval {1..32} is a good start as it has many small d's.
    # But we might want to spread them.
    # Let's start with a random set or a mix.
    
    def score(s):
        diffs = set()
        lst = sorted(s)
        for i in range(len(lst)):
            for j in range(i+1, len(lst)):
                d = lst[j] - lst[i]
                if d > 0:
                    if lst[i] + 2*d in s:
                        diffs.add(d)
        return len(diffs)
    
    def local_search(s, max_iter=1000):
        current_set = set(s)
        current_score = score(current_set)
        best_set = set(current_set)
        best_score = current_score
        
        for _ in range(max_iter):
            # Random swap
            a = random.choice(list(current_set))
            b = random.randint(1, 512)
            if b in current_set:
                continue
            new_set = set(current_set)
            new_set.remove(a)
            new_set.add(b)
            new_score = score(new_set)
            if new_score > current_score:
                current_set = new_set
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(current_set)
            # Optional: accept worse moves? No, keep it simple hill climbing for ILS steps.
            
        return best_set, best_score

    random.seed(42)
    
    # Start with interval
    current_set = set(range(1, 33))
    best_set, best_score = local_search(current_set, max_iter=2000)
    
    # Iterated Local Search
    for i in range(50):
        # Perturb the best set found so far
        perturbed_set = set(best_set)
        # Do a few random swaps to perturb
        for _ in range(5):
            a = random.choice(list(perturbed_set))
            b = random.randint(1, 512)
            if b not in perturbed_set:
                perturbed_set.remove(a)
                perturbed_set.add(b)
        
        # Local search from perturbed state
        s, sc = local_search(perturbed_set, max_iter=2000)
        if sc > best_score:
            best_score = sc
            best_set = s
            
    return sorted(best_set)
