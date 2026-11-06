import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 148.0
    
    def verify_coloring(n, colors):
        """Verify that a coloring has no red K4 or blue K5."""
        triples = list(combinations(range(n), 3))
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Check for red K4
        for comb4 in combinations(range(n), 4):
            a, b, c, d = comb4
            if (colors[triple_to_idx[(a,b,c)]] == 0 and 
                colors[triple_to_idx[(a,b,d)]] == 0 and 
                colors[triple_to_idx[(a,c,d)]] == 0 and 
                colors[triple_to_idx[(b,c,d)]] == 0):
                return False
        
        # Check for blue K5
        for comb5 in combinations(range(n), 5):
            a, b, c, d, e = comb5
            ts = [
                (a,b,c), (a,b,d), (a,b,e), (a,c,d), (a,c,e), 
                (a,d,e), (b,c,d), (b,c,e), (b,d,e), (c,d,e)
            ]
            if all(colors[triple_to_idx[t]] == 1 for t in ts):
                return False
        
        return True
    
    # Try to construct a valid coloring for n=34 using a deterministic approach
    # Based on known constructions for hypergraph Ramsey numbers
    n = 34
    triples = list(combinations(range(n), 3))
    colors = []
    
    # Use a construction based on modular arithmetic
    # Color triple (a,b,c) as red (0) if (a+b+c) mod 7 is in {0,1,2}, else blue (1)
    # This is a heuristic that may work for some sizes
    for (a, b, c) in triples:
        s = (a + b + c) % 7
        if s in {0, 1, 2}:
            colors.append(0)
        else:
            colors.append(1)
    
    if verify_coloring(n, colors):
        return {'n': n, 'colors': colors}
    
    # Try another construction: color based on product
    colors = []
    for (a, b, c) in triples:
        prod = (a * b * c) % 13
        if prod in {0, 1, 2, 3, 4}:
            colors.append(0)
        else:
            colors.append(1)
    
    if verify_coloring(n, colors):
        return {'n': n, 'colors': colors}
    
    # Try random initialization with local search for n=34
    if time.time() < deadline - 5.0:
        for attempt in range(10):
            if time.time() > deadline - 3.0:
                break
            
            colors = [random.randint(0, 1) for _ in triples]
            triple_to_idx = {t: i for i, t in enumerate(triples)}
            
            if verify_coloring(n, colors):
                return {'n': n, 'colors': colors}
            
            # Local search with incremental updates
            improved = True
            max_rounds = 50
            for rnd in range(max_rounds):
                if time.time() > deadline - 2.0:
                    break
                improved = False
                indices = list(range(len(colors)))
                random.shuffle(indices)
                
                for idx in indices:
                    if time.time() > deadline - 2.0:
                        break
                    
                    a, b, c = triples[idx]
                    old_color = colors[idx]
                    new_color = 1 - old_color
                    
                    delta = 0
                    
                    # Check K4s containing (a,b,c)
                    for d in range(n):
                        if d in (a, b, c):
                            continue
                        t2 = (a,b,d)
                        t3 = (a,c,d)
                        t4 = (b,c,d)
                        idx2 = triple_to_idx[t2]
                        idx3 = triple_to_idx[t3]
                        idx4 = triple_to_idx[t4]
                        
                        c2 = colors[idx2]
                        c3 = colors[idx3]
                        c4 = colors[idx4]
                        
                        is_red_before = (old_color == 0 and c2 == 0 and c3 == 0 and c4 == 0)
                        is_red_after = (new_color == 0 and c2 == 0 and c3 == 0 and c4 == 0)
                        
                        if is_red_before and not is_red_after:
                            delta -= 1
                        elif not is_red_before and is_red_after:
                            delta += 1
                    
                    # Check K5s containing (a,b,c)
                    others = [v for v in range(n) if v not in (a, b, c)]
                    for i in range(len(others)):
                        for j in range(i+1, len(others)):
                            d = others[i]
                            e = others[j]
                            
                            ts = [
                                (a,b,d), (a,b,e), (a,c,d), (a,c,e), (a,d,e),
                                (b,c,d), (b,c,e), (b,d,e), (c,d,e)
                            ]
                            
                            if old_color == 1:
                                if all(colors[triple_to_idx[t]] == 1 for t in ts):
                                    delta -= 1
                            elif old_color == 0:
                                if all(colors[triple_to_idx[t]] == 1 for t in ts):
                                    delta += 1
                    
                    if delta < 0:
                        colors[idx] = new_color
                        improved = True
                        if verify_coloring(n, colors):
                            return {'n': n, 'colors': colors}
                    
                    if not improved and rnd == 0:
                        break
                
                if not improved:
                    break
    
    # Try smaller n values
    for n in range(33, 9, -1):
        if time.time() > deadline - 2.0:
            break
        
        triples = list(combinations(range(n), 3))
        
        # Try simple constructions
        for rule in ['sum_mod7', 'sum_mod5', 'prod_mod13']:
            if time.time() > deadline - 2.0:
                break
            
            colors = []
            for (a, b, c) in triples:
                if rule == 'sum_mod7':
                    s = (a + b + c) % 7
                    colors.append(0 if s in {0, 1, 2} else 1)
                elif rule == 'sum_mod5':
                    s = (a + b + c) % 5
                    colors.append(0 if s in {0, 1} else 1)
                else:
                    prod = (a * b * c) % 13
                    colors.append(0 if prod in {0, 1, 2, 3, 4} else 1)
            
            if verify_coloring(n, colors):
                return {'n': n, 'colors': colors}
        
        # Random with local search
        if time.time() > deadline - 3.0:
            continue
        
        colors = [random.randint(0, 1) for _ in triples]
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        if verify_coloring(n, colors):
            return {'n': n, 'colors': colors}
        
        # Simple local search
        for rnd in range(20):
            if time.time() > deadline - 2.0:
                break
            improved = False
            indices = list(range(len(colors)))
            random.shuffle(indices)
            
            for idx in indices:
                if time.time() > deadline - 2.0:
                    break
                
                a, b, c = triples[idx]
                old_color = colors[idx]
                new_color = 1 - old_color
                
                delta = 0
                
                for d in range(n):
                    if d in (a, b, c):
                        continue
                    t2 = (a,b,d)
                    t3 = (a,c,d)
                    t4 = (b,c,d)
                    idx2 = triple_to_idx[t2]
                    idx3 = triple_to_idx[t3]
                    idx4 = triple_to_idx[t4]
                    
                    c2 = colors[idx2]
                    c3 = colors[idx3]
                    c4 = colors[idx4]
                    
                    is_red_before = (old_color == 0 and c2 == 0 and c3 == 0 and c4 == 0)
                    is_red_after = (new_color == 0 and c2 == 0 and c3 == 0 and c4 == 0)
                    
                    if is_red_before and not is_red_after:
                        delta -= 1
                    elif not is_red_before and is_red_after:
                        delta += 1
                
                others = [v for v in range(n) if v not in (a, b, c)]
                for i in range(len(others)):
                    for j in range(i+1, len(others)):
                        d = others[i]
                        e = others[j]
                        
                        ts = [
                            (a,b,d), (a,b,e), (a,c,d), (a,c,e), (a,d,e),
                            (b,c,d), (b,c,e), (b,d,e), (c,d,e)
                        ]
                        
                        if old_color == 1:
                            if all(colors[triple_to_idx[t]] == 1 for t in ts):
                                delta -= 1
                        elif old_color == 0:
                            if all(colors[triple_to_idx[t]] == 1 for t in ts):
                                delta += 1
                
                if delta < 0:
                    colors[idx] = new_color
                    improved = True
                    if verify_coloring(n, colors):
                        return {'n': n, 'colors': colors}
            
            if not improved:
                break
    
    # Fallback: return a valid small solution
    n = 9
    triples = list(combinations(range(n), 3))
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]
    
    if verify_coloring(n, colors):
        return {'n': n, 'colors': colors}
    
    # Last resort: all red (no blue K5 possible, but check red K4)
    n = 3
    triples = list(combinations(range(n), 3))
    colors = [0] * len(triples)
    
    return {'n': n, 'colors': colors}
