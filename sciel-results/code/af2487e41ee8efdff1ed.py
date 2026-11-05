import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    def build_structure(n):
        triples = list(combinations(range(n), 3))
        triple_index = {t: i for i, t in enumerate(triples)}
        
        sets_4 = list(combinations(range(n), 4))
        set4_triples = []
        for s in sets_4:
            ts = []
            for i in range(4):
                for j in range(i+1, 4):
                    for k in range(j+1, 4):
                        t = (s[i], s[j], s[k])
                        ts.append(triple_index[t])
            set4_triples.append(ts)
            
        sets_5 = list(combinations(range(n), 5))
        set5_triples = []
        for s in sets_5:
            ts = []
            for t in combinations(s, 3):
                ts.append(triple_index[t])
            set5_triples.append(ts)
            
        triple_to_set4 = [[] for _ in range(len(triples))]
        for i, ts in enumerate(set4_triples):
            for t_idx in ts:
                triple_to_set4[t_idx].append(i)
                
        triple_to_set5 = [[] for _ in range(len(triples))]
        for i, ts in enumerate(set5_triples):
            for t_idx in ts:
                triple_to_set5[t_idx].append(i)
                
        return {
            'n': n,
            'triples': triples,
            'set4_triples': set4_triples,
            'set5_triples': set5_triples,
            'triple_to_set4': triple_to_set4,
            'triple_to_set5': triple_to_set5,
            'num_triples': len(triples),
            'num_set4': len(sets_4),
            'num_set5': len(sets_5)
        }

    def count_violations(colors, struct):
        v4 = 0
        v5 = 0
        for ts in struct['set4_triples']:
            if colors[ts[0]] == 0 and colors[ts[1]] == 0 and colors[ts[2]] == 0 and colors[ts[3]] == 0:
                v4 += 1
        for ts in struct['set5_triples']:
            all_one = True
            for idx in ts:
                if colors[idx] == 0:
                    all_one = False
                    break
            if all_one:
                v5 += 1
        return v4 + v5

    def delta_violations(t_idx, new_color, colors, struct):
        old_color = colors[t_idx]
        if old_color == new_color:
            return 0
        d_v = 0
        for s4_idx in struct['triple_to_set4'][t_idx]:
            ts = struct['set4_triples'][s4_idx]
            cnt0_before = 0
            for idx in ts:
                if idx == t_idx:
                    if old_color == 0: cnt0_before += 1
                else:
                    if colors[idx] == 0: cnt0_before += 1
            cnt0_after = cnt0_before
            if old_color == 0 and new_color == 1:
                cnt0_after -= 1
            elif old_color == 1 and new_color == 0:
                cnt0_after += 1
            viol_before = 1 if cnt0_before == 4 else 0
            viol_after = 1 if cnt0_after == 4 else 0
            d_v += (viol_after - viol_before)
            
        for s5_idx in struct['triple_to_set5'][t_idx]:
            ts = struct['set5_triples'][s5_idx]
            cnt1_before = 0
            for idx in ts:
                if idx == t_idx:
                    if old_color == 1: cnt1_before += 1
                else:
                    if colors[idx] == 1: cnt1_before += 1
            cnt1_after = cnt1_before
            if old_color == 1 and new_color == 0:
                cnt1_after -= 1
            elif old_color == 0 and new_color == 1:
                cnt1_after += 1
            viol_before = 1 if cnt1_before == 10 else 0
            viol_after = 1 if cnt1_after == 10 else 0
            d_v += (viol_after - viol_before)
        return d_v

    best_n = 9
    best_colors = None

    # Bootstrap with n=10
    current_n = 10
    struct = build_structure(current_n)
    colors = np.random.randint(0, 2, size=struct['num_triples']).astype(int)
    
    # List of triples that involve the new vertex (for restricted search)
    # When extending from n-1 to n, new vertex is n-1.
    # Triples with n-1 are those where max(triple) == n-1.
    
    def get_new_triples(struct, n):
        """Indices of triples involving vertex n-1"""
        new_triples = []
        for i, t in enumerate(struct['triples']):
            if t[2] == n - 1:
                new_triples.append(i)
        return new_triples

    while time.time() < deadline - 5.0:
        # Determine candidate triples to flip
        # If we just extended, restrict to new triples.
        # If we are stuck, allow all triples.
        
        # For the first n=10, we have no previous, so all are "new".
        # Let's track if we are in "extension mode".
        # Actually, simpler: Always restrict to new triples first. If stuck, expand to all.
        
        new_triples = get_new_triples(struct, current_n)
        if not new_triples:
            new_triples = list(range(struct['num_triples']))
            
        current_viol = count_violations(colors, struct)
        
        if current_viol == 0:
            best_n = current_n
            best_colors = colors.copy()
            # Extend to next n
            current_n += 1
            if current_n > 35: break
            
            # Extend colors
            old_struct = struct
            struct = build_structure(current_n)
            new_colors = np.zeros(struct['num_triples'], dtype=int)
            old_triple_set = {t: i for i, t in enumerate(old_struct['triples'])}
            for i, t in enumerate(struct['triples']):
                if t[2] < current_n - 1:
                    new_colors[i] = best_colors[old_triple_set[t]]
                else:
                    new_colors[i] = random.randint(0, 1)
            colors = new_colors
            continue

        # Search
        # Try to fix violations using only new_triples first
        # If no improvement in 1000 steps, allow all triples
        
        allow_all = False
        no_improve = 0
        
        for step in range(2000):
            if time.time() > deadline - 2.0:
                break
                
            if not allow_all and no_improve > 500:
                allow_all = True
                no_improve = 0
                
            candidates = range(struct['num_triples']) if allow_all else new_triples
            t_idx = random.choice(list(candidates))
            
            new_color = 1 - colors[t_idx]
            d_v = delta_violations(t_idx, new_color, colors, struct)
            
            if d_v < 0:
                colors[t_idx] = new_color
                current_viol += d_v
                no_improve = 0
            elif d_v == 0:
                if random.random() < 0.5:
                    colors[t_idx] = new_color
                    no_improve = 0
            else:
                no_improve += 1
                # SA acceptance
                if random.random() < np.exp(-d_v / 0.1):
                    colors[t_idx] = new_color
                    current_viol += d_v
                    no_improve = 0

            if current_viol == 0:
                break

    if best_colors is not None:
        return {'n': best_n, 'colors': best_colors.tolist()}
    else:
        return {'n': 9, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]}
# EVOLVE-BLOCK-END
