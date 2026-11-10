import time
import random

def construct():
    N = 51975
    divisors = []
    for i in range(3, N + 1, 2):
        if N % i == 0:
            divisors.append(i)
    
    indices = {}
    for m in divisors:
        indices[m] = [[] for _ in range(m)]
        for r in range(m):
            indices[m][r] = list(range(r, N, m))
            
    best_score = N
    best_assignment = {}
    
    deadline = time.time() + 145
    
    def greedy_asc():
        covered = bytearray(N)
        assignment = {}
        for m in sorted(divisors):
            best_r = 0
            max_cov = -1
            for r in range(m):
                cnt = 0
                for x in indices[m][r]:
                    if covered[x] == 0:
                        cnt += 1
                if cnt > max_cov:
                    max_cov = cnt
                    best_r = r
            assignment[m] = best_r
            for x in indices[m][best_r]:
                covered[x] = 1
        return assignment

    def greedy_desc():
        covered = bytearray(N)
        assignment = {}
        for m in sorted(divisors, reverse=True):
            best_r = 0
            max_cov = -1
            for r in range(m):
                cnt = 0
                for x in indices[m][r]:
                    if covered[x] == 0:
                        cnt += 1
                if cnt > max_cov:
                    max_cov = cnt
                    best_r = r
            assignment[m] = best_r
            for x in indices[m][best_r]:
                covered[x] = 1
        return assignment

    def random_init():
        return {m: random.randint(0, m-1) for m in divisors}

    def local_search(assignment, max_passes=500):
        counts = [0] * N
        for m in divisors:
            r = assignment[m]
            for x in indices[m][r]:
                counts[x] += 1
        
        score = sum(1 for c in counts if c == 0)
        
        passes = 0
        while passes < max_passes:
            if time.time() > deadline:
                break
            improved = False
            passes += 1
            
            # First Improvement
            for m in divisors:
                if time.time() > deadline:
                    break
                r_old = assignment[m]
                # Shuffle residues to avoid bias? Or just iterate.
                # Iterating 0..m-1 is fine.
                for r_new in range(m):
                    if r_new == r_old:
                        continue
                    
                    lost = 0
                    for x in indices[m][r_old]:
                        if counts[x] == 1:
                            lost += 1
                            
                    gained = 0
                    for x in indices[m][r_new]:
                        if counts[x] == 0:
                            gained += 1
                            
                    delta = lost - gained
                    if delta < 0:
                        # Apply move
                        for x in indices[m][r_old]:
                            counts[x] -= 1
                        for x in indices[m][r_new]:
                            counts[x] += 1
                        score += delta
                        assignment[m] = r_new
                        improved = True
                        break # Move to next m
                
        return assignment, score

    for _ in range(500):
        if time.time() > deadline:
            break
            
        r = random.random()
        if r < 0.4:
            init = greedy_asc()
        elif r < 0.8:
            init = greedy_desc()
        else:
            init = random_init()
            
        res, score = local_search(init)
        
        if score < best_score:
            best_score = score
            best_assignment = res
            
        if best_score == 0:
            break

    if not best_assignment:
        best_assignment = greedy_asc()
        
    classes = []
    for m in divisors:
        classes.append([m, best_assignment[m]])
        
    return classes
