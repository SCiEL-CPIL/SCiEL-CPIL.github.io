"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def pebble(dag, fast_memory_budget):
    def _extract_structure(dag):
        parents = None
        inputs = None
        outputs = None
        n = None
        
        if isinstance(dag, dict):
            if 'parents' in dag:
                parents = dag['parents']
            if 'inputs' in dag:
                inputs = set(dag['inputs'])
            elif 'input' in dag:
                inputs = set(dag['input'])
            if 'outputs' in dag:
                outputs = set(dag['outputs'])
            elif 'output' in dag:
                outputs = set(dag['output'])
            if 'n' in dag:
                n = dag['n']
            elif 'num_vertices' in dag:
                n = dag['num_vertices']
            elif 'vertices' in dag:
                n = len(dag['vertices'])
        else:
            for attr in ['parents', 'parent_list', 'preds']:
                if hasattr(dag, attr):
                    parents = getattr(dag, attr)
                    break
            for attr in ['inputs', 'input', 'input_vertices', 'sources']:
                if hasattr(dag, attr):
                    inputs = set(getattr(dag, attr))
                    break
            for attr in ['outputs', 'output', 'output_vertices', 'sinks']:
                if hasattr(dag, attr):
                    outputs = set(getattr(dag, attr))
                    break
            for attr in ['n', 'num_vertices', 'num_nodes', 'size']:
                if hasattr(dag, attr):
                    n = getattr(dag, attr)
                    break
        
        if n is None:
            if parents is not None:
                if isinstance(parents, list):
                    n = len(parents)
                elif isinstance(parents, dict):
                    n = max(parents.keys()) + 1 if parents else 0
            else:
                n = 0
        
        if inputs is None:
            inputs = set()
            if parents is not None:
                for v in range(n):
                    if isinstance(parents, list):
                        p = parents[v] if v < len(parents) else []
                    elif isinstance(parents, dict):
                        p = parents.get(v, [])
                    else:
                        p = []
                    if not p:
                        inputs.add(v)
        
        if outputs is None:
            children = {v: [] for v in range(n)}
            if parents is not None:
                for v in range(n):
                    if isinstance(parents, list):
                        p = parents[v] if v < len(parents) else []
                    elif isinstance(parents, dict):
                        p = parents.get(v, [])
                    else:
                        p = []
                    for par in p:
                        children[par].append(v)
            outputs = set(v for v in range(n) if not children[v])
        
        return n, parents, inputs, outputs
    
    n, parents, inputs, outputs = _extract_structure(dag)
    
    children = {v: [] for v in range(n)}
    for v in range(n):
        if isinstance(parents, list):
            p = parents[v] if v < len(parents) else []
        elif isinstance(parents, dict):
            p = parents.get(v, [])
        else:
            p = []
        for par in p:
            children[par].append(v)
    
    remaining_children = {v: len(children[v]) for v in range(n)}
    
    ops = []
    in_fast = set()
    
    for v in range(n):
        if isinstance(parents, list):
            p = parents[v] if v < len(parents) else []
        elif isinstance(parents, dict):
            p = parents.get(v, [])
        else:
            p = []
        
        if not p:
            ops.append(['load', v])
            in_fast.add(v)
        else:
            for par in p:
                if par not in in_fast:
                    ops.append(['load', par])
                    in_fast.add(par)
            
            ops.append(['compute', v])
            in_fast.add(v)
            
            for par in p:
                remaining_children[par] -= 1
                # Don't delete if it's an output (we need to store it later)
                if remaining_children[par] == 0 and par in in_fast and par not in outputs:
                    ops.append(['delete', par])
                    in_fast.discard(par)
    
    for v in sorted(outputs):
        if v in in_fast:
            ops.append(['store', v])
        else:
            ops.append(['load', v])
            ops.append(['store', v])
    
    if len(ops) > 20000:
        ops = ops[:20000]
    
    return ops
# EVOLVE-BLOCK-END
