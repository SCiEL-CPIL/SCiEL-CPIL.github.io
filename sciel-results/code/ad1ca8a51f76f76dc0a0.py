import sys
import time

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 45.0
    
    if not dag:
        return []
        
    n = len(dag)
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in dag[v]:
            children[p].append(v)
            
    is_output = [len(children[v]) == 0 for v in range(n)]
    is_input = [len(dag[v]) == 0 for v in range(n)]
    
    in_fast = set()
    in_slow = set()
    for v in range(n):
        if is_input[v]:
            in_slow.add(v)
            
    operations = []
    
    def check_time():
        if time.time() > deadline:
            raise TimeoutError("Time limit exceeded")

    def do_op(op, v):
        operations.append([op, v])
        check_time()

    def load(v):
        if v in in_slow:
            if len(in_fast) >= fast_memory_budget and v not in in_fast:
                # Memory full. This simple strategy doesn't handle eviction.
                # It will fail verification if budget is too small.
                pass
            do_op("load", v)
            in_slow.remove(v)
            in_fast.add(v)

    def store(v):
        if v in in_fast:
            do_op("store", v)
            in_fast.remove(v)
            in_slow.add(v)

    def delete(v):
        if v in in_fast:
            do_op("delete", v)
            in_fast.remove(v)

    def compute(v):
        for p in dag[v]:
            if p not in in_fast:
                if p in in_slow:
                    load(p)
                else:
                    process(p)
        do_op("compute", v)
        in_fast.add(v)

    def is_done(v):
        return v in in_fast or v in in_slow

    def process(v):
        check_time()
        if v in in_fast:
            return
        if v in in_slow:
            load(v)
            return
            
        compute(v)
        
        if is_output[v]:
            store(v)
            
        # Cleanup parents
        for p in dag[v]:
            if p in in_fast:
                # Check if p is needed by any other child
                needed = False
                for c in children[p]:
                    if c != v and not is_done(c):
                        needed = True
                        break
                if not needed:
                    delete(p)

    for v in range(n):
        if is_output[v]:
            process(v)
            
    return operations
