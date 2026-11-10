import itertools
import time

def construct():
    # Fallback for safety
    def get_fallback():
        n = 9
        colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in itertools.combinations(range(n), 3)]
        return {'n': n, 'colors': colors}

    def is_valid_coloring(n, colors):
        # Map triple to index
        triples = list(itertools.combinations(range(n), 3))
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Check Red K4
        for quad in itertools.combinations(range(n), 4):
            # 4 triples in quad
            t1 = (quad[0], quad[1], quad[2])
            t2 = (quad[0], quad[1], quad[3])
            t3 = (quad[0], quad[2], quad[3])
            t4 = (quad[1], quad[2], quad[3])
            if colors[triple_to_idx[t1]] == 0 and \
               colors[triple_to_idx[t2]] == 0 and \
               colors[triple_to_idx[t3]] == 0 and \
               colors[triple_to_idx[t4]] == 0:
                return False
        
        # Check Blue K5
        for quint in itertools.combinations(range(n), 5):
            all_blue = True
            for tri in itertools.combinations(quint, 3):
                if colors[triple_to_idx[tri]] != 1:
                    all_blue = False
                    break
            if all_blue:
                return False
        return True

    def generate_coloring(n, rule_func):
        colors = []
        for a, b, c in itertools.combinations(range(n), 3):
            colors.append(rule_func(a, b, c))
        return colors

    # Define candidate rules
    # rule_func(a, b, c) returns 0 or 1
    rules = []
    
    # Rule 1: Parity of sum
    rules.append(lambda a,b,c: (a+b+c) % 2)
    # Rule 2: Parity of span
    rules.append(lambda a,b,c: (c-a) % 2)
    # Rule 3: Parity of first diff
    rules.append(lambda a,b,c: (b-a) % 2)
    # Rule 4: Parity of second diff
    rules.append(lambda a,b,c: (c-b) % 2)
    # Rule 5: XOR of parities of diffs
    rules.append(lambda a,b,c: ((b-a) % 2) ^ ((c-b) % 2))
    # Rule 6: Mod 3 of sum
    rules.append(lambda a,b,c: 0 if (a+b+c) % 3 == 0 else 1)
    # Rule 7: Mod 3 of first diff
    rules.append(lambda a,b,c: 0 if (b-a) % 3 == 0 else 1)
    # Rule 8: Mod 3 of span
    rules.append(lambda a,b,c: 0 if (c-a) % 3 == 0 else 1)
    # Rule 9: Mod 5 of sum
    rules.append(lambda a,b,c: 0 if (a+b+c) % 5 == 0 else 1)
    # Rule 10: Mod 5 of first diff
    rules.append(lambda a,b,c: 0 if (b-a) % 5 == 0 else 1)
    # Rule 11: Mod 5 of span
    rules.append(lambda a,b,c: 0 if (c-a) % 5 == 0 else 1)
    # Rule 12: (b-a) % 4 < 2
    rules.append(lambda a,b,c: 0 if (b-a) % 4 < 2 else 1)
    # Rule 13: (c-b) % 4 < 2
    rules.append(lambda a,b,c: 0 if (c-b) % 4 < 2 else 1)
    # Rule 14: (c-a) % 4 < 2
    rules.append(lambda a,b,c: 0 if (c-a) % 4 < 2 else 1)
    # Rule 15: (a+b+c) % 7 < 3
    rules.append(lambda a,b,c: 0 if (a+b+c) % 7 < 3 else 1)
    # Rule 16: (b-a) % 7 < 3
    rules.append(lambda a,b,c: 0 if (b-a) % 7 < 3 else 1)
    # Rule 17: (c-b) % 7 < 3
    rules.append(lambda a,b,c: 0 if (c-b) % 7 < 3 else 1)
    # Rule 18: (c-a) % 7 < 3
    rules.append(lambda a,b,c: 0 if (c-a) % 7 < 3 else 1)
    # Rule 19: (b-a + c-b) % 6 < 3
    rules.append(lambda a,b,c: 0 if (c-a) % 6 < 3 else 1)
    # Rule 20: (b-a) % 6 < 3
    rules.append(lambda a,b,c: 0 if (b-a) % 6 < 3 else 1)

    # Try to find a valid coloring for n from 34 down to 10
    start_time = time.time()
    time_limit = 110
    
    for n in range(34, 9, -1):
        if time.time() - start_time > time_limit:
            break
        for rule in rules:
            if time.time() - start_time > time_limit:
                break
            try:
                colors = generate_coloring(n, rule)
                if is_valid_coloring(n, colors):
                    return {'n': n, 'colors': colors}
            except Exception:
                continue
    
    # Fallback
    return get_fallback()
