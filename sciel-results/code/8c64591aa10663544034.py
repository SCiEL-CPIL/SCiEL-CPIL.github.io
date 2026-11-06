"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    
    vertices = dag['vertices']
    parents = dag['parents']
    inputs = dag['inputs']
    outputs = dag['outputs']
    n = len(vertices)
    k = fast_memory_budget
    
    if n == 0:
        return []
    
    layer = [0] * n
    input_set = set(inputs)
    for i in range(n):
        if i in input_set:
            layer[i] = 0
        elif parents[i]:
            layer[i] = 1 + max(layer[p] for p in parents[i])
        else:
            layer[i] = 0
    
    max_layer = max(layer) if n > 0 else 0
    layers = [[] for _ in range(max_layer + 1)]
    for i in range(n):
        layers[layer[i]].append(i)
    
    L = len(layers)
    
    is_layered = True
    for i in range(n):
        if layer[i] > 0:
            for p in parents[i]:
                if layer[p] != layer[i] - 1:
                    is_layered = False
                    break
        if not is_layered:
            break
    
    if is_layered and L > 1:
        schedule = _layered_schedule(layers, parents, inputs, outputs, k, n)
    else:
        schedule = _generic_schedule(parents, inputs, outputs, k, n)
    
    if len(schedule) > 20000:
        schedule = schedule[:20000]
    
    return schedule


def _layered_schedule(layers, parents, inputs, outputs, k, n):
    L = len(layers)
    widths = [len(layers[l]) for l in range(L)]
    w = max(widths) if widths else 1
    
    if w == 0 or L == 0:
        return []
    
    b = max(1, k // w)
    
    if b >= L:
        schedule = []
        for l in range(L):
            for v in layers[l]:
                schedule.append(['compute', v])
        return schedule
    
    c = b
    schedule = []
    in_mem = set(inputs)
    disk = set()
    output_set = set(outputs)
    
    for l in range(L):
        for v in layers[l]:
            for p in parents[v]:
                if p not in in_mem:
                    if p in disk:
                        schedule.append(['load', p])
                        in_mem.add(p)
                        disk.discard(p)
        
        for v in layers[l]:
            schedule.append(['compute', v])
            in_mem.add(v)
        
        if l % c == 0 and l < L - 1:
            for v in layers[l]:
                if v not in output_set:
                    schedule.append(['store', v])
                    disk.add(v)
                    in_mem.discard(v)
        else:
            for ll in range(l - 1, -1, -1):
                if ll % c == 0:
                    continue
                for v in layers[ll]:
                    if v in in_mem and v not in output_set:
                        schedule.append(['delete', v])
                        in_mem.discard(v)
        
        while len(in_mem) > k:
            for ll in range(l, -1, -1):
                if ll % c == 0:
                    continue
                for v in layers[ll]:
                    if v in in_mem and v not in output_set:
                        schedule.append(['delete', v])
                        in_mem.discard(v)
                        if len(in_mem) <= k:
                            break
                if len(in_mem) <= k:
                    break
    
    for v in outputs:
        if v not in in_mem:
            if v in disk:
                schedule.append(['load', v])
                in_mem.add(v)
                disk.discard(v)
            else:
                _recompute(v, parents, in_mem, disk, schedule, output_set)
    
    return schedule


def _recompute(v, parents, in_mem, disk, schedule, output_set):
    for p in parents[v]:
        if p not in in_mem:
            if p in disk:
                schedule.append(['load', p])
                in_mem.add(p)
                disk.discard(p)
            else:
                _recompute(p, parents, in_mem, disk, schedule, output_set)
    schedule.append(['compute', v])
    in_mem.add(v)


def _generic_schedule(parents, inputs, outputs, k, n):
    schedule = []
    in_mem = set(inputs)
    output_set = set(outputs)
    
    for v in range(n):
        for p in parents[v]:
            if p not in in_mem:
                schedule.append(['load', p])
                in_mem.add(p)
        
        schedule.append(['compute', v])
        in_mem.add(v)
        
        while len(in_mem) > k:
            deleted = False
            for u in range(v - 1, -1, -1):
                if u in in_mem and u not in output_set:
                    schedule.append(['delete', u])
                    in_mem.discard(u)
                    deleted = True
                    break
            if not deleted:
                break
    
    for v in outputs:
        if v not in in_mem:
            schedule.append(['load', v])
            in_mem.add(v)
    
    return schedule
# EVOLVE-BLOCK-END
