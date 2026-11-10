import time
from itertools import combinations

def construct():
    def greedy_construct(n):
        triples = list(combinations(range(n), 3))
        idx_map = {t: i for i, t in enumerate(triples)}
        colors = [0] * len(triples)
        
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    idx = idx_map[(a,b,c)]
                    
                    red_forbidden = False
                    blue_forbidden = False
                    
                    # Check Red Forbidden
                    red_threats = 0
                    for d in range(a):
                        c1 = colors[idx_map[(d,a,b)]]
                        c2 = colors[idx_map[(d,a,c)]]
                        c3 = colors[idx_map[(d,b,c)]]
                        if c1 == 0 and c2 == 0 and c3 == 0:
                            red_forbidden = True
                            break
                        elif c1 == 0 and c2 == 0 and c3 == 1:
                            red_threats += 1
                        elif c1 == 0 and c2 == 1 and c3 == 0:
                            red_threats += 1
                        elif c1 == 1 and c2 == 0 and c3 == 0:
                            red_threats += 1
                            
                    # Check Blue Forbidden
                    # Pre-compute d's that are "blue-compatible" with a,b,c
                    valid_ds = []
                    for d in range(a):
                        c1 = colors[idx_map[(d,a,b)]]
                        c2 = colors[idx_map[(d,a,c)]]
                        c3 = colors[idx_map[(d,b,c)]]
                        if c1 == 1 and c2 == 1 and c3 == 1:
                            valid_ds.append(d)
                            
                    blue_threats = 0
                    for i in range(len(valid_ds)):
                        d = valid_ds[i]
                        for j in range(i):
                            e = valid_ds[j]
                            # Check edges between d,e and a,b,c
                            # (d,e,a), (d,e,b), (d,e,c)
                            c1 = colors[idx_map[(d,e,a)]]
                            c2 = colors[idx_map[(d,e,b)]]
                            c3 = colors[idx_map[(d,e,c)]]
                            if c1 == 1 and c2 == 1 and c3 == 1:
                                blue_forbidden = True
                                break
                            # Count threats: if 2 of these 3 are Blue, it's a threat
                            # Note: The 3 edges (d,a,b), (d,a,c), (d,b,c) are already known Blue.
                            # So we have 3 Blue edges.
                            # If we add (a,b,c) as Blue, we have 4.
                            # We need 7 total. So we need 3 more from (d,e,a), (d,e,b), (d,e,c).
                            # If 2 are Blue, we are 1 away from Forbidden? No.
                            # Forbidden means ALL 7 are Blue.
                            # So if (d,e,a), (d,e,b), (d,e,c) are ALL Blue, then Forbidden.
                            # Threat: If 2 are Blue, then if the 3rd becomes Blue, it's Forbidden.
                            # But the 3rd is fixed.
                            # So "Threat" here is weak.
                            # Let's just count how many of the 3 are Blue.
                            blue_count = (c1 == 1) + (c2 == 1) + (c3 == 1)
                            if blue_count == 2:
                                blue_threats += 1
                        if blue_forbidden:
                            break
                            
                    if red_forbidden and blue_forbidden:
                        return None
                    elif red_forbidden:
                        colors[idx] = 1
                    elif blue_forbidden:
                        colors[idx] = 0
                    else:
                        # Heuristic: Choose color with fewer threats
                        if red_threats < blue_threats:
                            colors[idx] = 0
                        else:
                            colors[idx] = 1
                            
        return colors

    res = greedy_construct(34)
    if res is not None:
        return {'n': 34, 'colors': res}
        
    # If it fails, try n=33
    res = greedy_construct(33)
    if res is not None:
        return {'n': 33, 'colors': res}
        
    return {'n': 1, 'colors': [0]}
