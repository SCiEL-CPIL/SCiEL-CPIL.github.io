# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    deadline = time.time() + 80
    N = 128
    elems = list(range(1, N + 1))
    
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    def is_bad_triple(a, b, c):
        return gcd_table[a][b] == gcd_table[a][c] == gcd_table[b][c]
    
    def can_add(x, sel_list):
        for i in range(len(sel_list)):
            a = sel_list[i]
            for j in range(i + 1, len(sel_list)):
                b = sel_list[j]
                if is_bad_triple(x, a, b):
                    return False
        return True
    
    # Phase 1: Start with all, iteratively remove worst
    selected = set(elems)
    while time.time() < deadline - 30:
        worst = None
        max_conf = 0
        sel_list = list(selected)
        for x in sel_list:
            conf = 0
            for i in range(len(sel_list)):
                a = sel_list[i]
                if a == x:
                    continue
                for j in range(i + 1, len(sel_list)):
                    b = sel_list[j]
                    if b == x:
                        continue
                    if is_bad_triple(x, a, b):
                        conf += 1
            if conf > max_conf:
                max_conf = conf
                worst = x
        if worst is None or max_conf == 0:
            break
        selected.remove(worst)
    
    # Phase 2: Greedy add-back
    removed = set(elems) - selected
    for x in list(removed):
        if time.time() > deadline - 5:
            break
        if can_add(x, list(selected)):
            selected.add(x)
    
    return list(selected)
# EVOLVE-BLOCK-END
