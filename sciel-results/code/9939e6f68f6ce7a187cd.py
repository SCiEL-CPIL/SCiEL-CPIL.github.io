import random
import itertools

def construct():
    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    
    # Precompute: for each 6-subset, all partitions and their 8 transversals
    six_data = []  # list of (six_tuple, frozenset_of_8_transversals)
    for six in itertools.combinations(range(n), 6):
        a, b, c, d, e, f = six
        partitions = [
            ((a,b),(c,d),(e,f)),
            ((a,b),(c,e),(d,f)),
            ((a,b),(c,f),(d,e)),
            ((a,c),(b,d),(e,f)),
            ((a,c),(b,e),(d,f)),
            ((a,c),(b,f),(d,e)),
            ((a,d),(b,c),(e,f)),
            ((a,d),(b,e),(c,f)),
            ((a,d),(b,f),(c,e)),
            ((a,e),(b,c),(d,f)),
            ((a,e),(b,d),(c,f)),
            ((a,e),(b,f),(c,d)),
            ((a,f),(b,c),(d,e)),
            ((a,f),(b,d),(c,e)),
            ((a,f),(b,e),(c,d)),
        ]
        for p1, p2, p3 in partitions:
            trans = set()
            for x in p1:
                for y in p2:
                    for z in p3:
                        trans.add((x, y, z))
            six_data.append((six, frozenset(trans)))
    
    # Map each triple to the list of (six, trans_frozenset) where trans contains the triple
    triple_to_configs = {t: [] for t in all_triples}
    for six, trans in six_data:
        six_set = set(six)
        for t in all_triples:
            if all(v in six_set for v in t) and t in trans:
                triple_to_configs[t].append(trans)
    
    def check_add(triple, current):
        for trans in triple_to_configs[triple]:
            if all(t in current or t == triple for t in trans):
                return True
        return False
    
    best = None
    best_len = 0
    
    for trial in range(50):
        random.seed(trial * 31337 + 42)
        order = all_triples[:]
        random.shuffle(order)
        cur = set()
        for t in order:
            if not check_add(t, cur):
                cur.add(t)
        if len(cur) > best_len:
            best_len = len(cur)
            best = cur
    
    # Final validation
    final = set(best)
    for six, trans in six_data:
        if all(t in final for t in trans):
            raise AssertionError("K(2,2,2) detected")
    
    return [[a, b, c] for a, b, c in best]
