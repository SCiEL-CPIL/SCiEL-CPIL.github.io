import time

# EVOLVE-BLOCK-START

def _children(n, parents):
    ch = [[] for _ in range(n)]
    for j in range(n):
        for p in parents[j]:
            ch[p].append(j)
    return ch

def _is_output(dag):
    n = len(dag['parents'])
    outs = dag.get('outputs')
    if outs is None:
        return [True]*n
    s = set(outs)
    return [v in s for v in range(n)]

def _greedy(dag):
    n = len(dag['parents'])
    B = max(1, int(dag['fast_memory_budget']))
    parents = dag['parents']
    ch = _children(n, parents)
    is_output = _is_output(dag)
    rem = [len(ch[v]) for v in range(n)]
    state = [0]*n  # 0=white,1=blue
    for v in range(n):
        if len(parents[v]) == 0:
            state[v] = 1
    def mem_count():
        return sum(1 for i in range(n) if state[i] == 1)
    out = []
    for v in range(n):
        for p in parents[v]:
            if state[p] == 0:
                if mem_count() >= B:
                    best = None; best_rem = 10**9
                    for i in range(n):
                        if state[i] == 1 and not is_output[i] and i != v:
                            if rem[i] < best_rem:
                                best_rem = rem[i]; best = i
                    if best is None:
                        for i in range(n):
                            if state[i] == 1 and i != v:
                                best = i; break
                    if best is not None:
                        state[best] = 0
                        out.append(['delete', best])
                state[p] = 1
                out.append(['load', p])
        out.append(['compute', v])
        state[v] = 1
        out.append(['store', v])
        for c in ch[v]:
            rem[c] -= 1
    return out

def pebble(dag, fast_memory_budget):
    t0 = time.time()
    try:
        return _greedy(dag)
    except Exception:
        n = len(dag['parents'])
        out = []
        for v in range(n):
            ps = dag['parents'][v]
            if len(ps) == 0:
                out.append(['load', v])
            else:
                for p in ps:
                    out.append(['load', p])
                out.append(['compute', v])
        return out

# EVOLVE-BLOCK-END
