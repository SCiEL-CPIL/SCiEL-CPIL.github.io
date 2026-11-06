import random
from itertools import combinations

def construct():
    n = 34
    random.seed(42)
    
    triples = list(combinations(range(n), 3))
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    num_triples = len(triples)
    
    K4s = list(combinations(range(n), 4))
    K5s = list(combinations(range(n), 5))
    num_K4s = len(K4s)
    num_K5s = len(K5s)
    
    K4_triples = []
    for k4 in K4s:
        tris = []
        for a, b, c in combinations(k4, 3):
            tris.append(triple_to_idx[(a, b, c)])
        K4_triples.append(tris)
        
    K5_triples = []
    for k5 in K5s:
        tris = []
        for a, b, c in combinations(k5, 3):
            tris.append(triple_to_idx[(a, b, c)])
        K5_triples.append(tris)
        
    triple_to_K4s = [[] for _ in range(num_triples)]
    for i, tris in enumerate(K4_triples):
        for t in tris:
            triple_to_K4s[t].append(i)
            
    triple_to_K5s = [[] for _ in range(num_triples)]
    for i, tris in enumerate(K5_triples):
        for t in tris:
            triple_to_K5s[t].append(i)
            
    colors = [random.randint(0, 1) for _ in range(num_triples)]
    
    red_K4_counts = [0] * num_K4s
    blue_K5_counts = [0] * num_K5s
    
    # Initialize counts
    for i in range(num_K4s):
        for t in K4_triples[i]:
            if colors[t] == 0:
                red_K4_counts[i] += 1
                
    for i in range(num_K5s):
        for t in K5_triples[i]:
            if colors[t] == 1:
                blue_K5_counts[i] += 1
                
    bad_red_K4s = set(i for i in range(num_K4s) if red_K4_counts[i] == 4)
    bad_blue_K5s = set(i for i in range(num_K5s) if blue_K5_counts[i] == 10)
    
    max_iterations = 10000
    for _ in range(max_iterations):
        if not bad_red_K4s and not bad_blue_K5s:
            break
            
        if bad_red_K4s:
            if random.random() < 0.5 or not bad_blue_K5s:
                k4_idx = random.choice(list(bad_red_K4s))
                t_idx = random.choice(K4_triples[k4_idx])
            else:
                k5_idx = random.choice(list(bad_blue_K5s))
                t_idx = random.choice(K5_triples[k5_idx])
        else:
            k5_idx = random.choice(list(bad_blue_K5s))
            t_idx = random.choice(K5_triples[k5_idx])
            
        # Flip t_idx
        old_color = colors[t_idx]
        new_color = 1 - old_color
        colors[t_idx] = new_color
        
        # Update counts
        # If old was 0 (red), new is 1 (blue)
        # red_K4_counts decrease by 1 for all K4s containing t_idx
        # blue_K5_counts increase by 1 for all K5s containing t_idx
        if old_color == 0:
            for k4_i in triple_to_K4s[t_idx]:
                red_K4_counts[k4_i] -= 1
                if red_K4_counts[k4_i] == 4:
                    bad_red_K4s.add(k4_i)
                else:
                    bad_red_K4s.discard(k4_i)
                    
            for k5_i in triple_to_K5s[t_idx]:
                blue_K5_counts[k5_i] += 1
                if blue_K5_counts[k5_i] == 10:
                    bad_blue_K5s.add(k5_i)
                else:
                    bad_blue_K5s.discard(k5_i)
        else:
            for k4_i in triple_to_K4s[t_idx]:
                red_K4_counts[k4_i] += 1
                if red_K4_counts[k4_i] == 4:
                    bad_red_K4s.add(k4_i)
                else:
                    bad_red_K4s.discard(k4_i)
                    
            for k5_i in triple_to_K5s[t_idx]:
                blue_K5_counts[k5_i] -= 1
                if blue_K5_counts[k5_i] == 10:
                    bad_blue_K5s.add(k5_i)
                else:
                    bad_blue_K5s.discard(k5_i)
                    
    return {'n': n, 'colors': colors}
