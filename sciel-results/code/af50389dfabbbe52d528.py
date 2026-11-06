<<<<<<< SEARCH
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin before 60s limit
    
    vertices = dag['vertices']
    parents = dag['parents']
    inputs = set(dag['inputs'])
    outputs = set(dag['outputs'])
    
    n = len(vertices)
    
    # Build children map
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            if p in children:
                children[p].append(v)
=======
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin before 60s limit
    
    try:
        vertices, parents, inputs, outputs = _extract_dag_info(dag)
    except Exception as e:
        # Fallback if extraction fails, try to return empty or minimal
        # But we need to know vertices to do anything.
        # If this fails, the program will crash. 
        # Let's assume the keys are standard or the extractor is robust enough.
        raise e

    inputs = set(inputs)
    outputs = set(outputs)
    
    n = len(vertices)
    
    # Build children map
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            if p in children:
                children[p].append(v)
>>>>>>> REPLACE

I also need to define `_extract_dag_info` before `pebble`.

