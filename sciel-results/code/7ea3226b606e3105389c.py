<<<<<<< SEARCH
def _get_dag_info(dag):
    """Extract vertices, parents, outputs from dag object."""
    if hasattr(dag, 'vertices'):
        vertices = list(dag.vertices)
    elif isinstance(dag, dict) and 'vertices' in dag:
        vertices = list(dag['vertices'])
    else:
        raise ValueError("Cannot determine vertices")
    
    if hasattr(dag, 'parents'):
        parents = dict(dag.parents)
    elif isinstance(dag, dict) and 'parents' in dag:
        parents = dict(dag['parents'])
    else:
        parents = {}
    
    if hasattr(dag, 'outputs'):
        outputs = list(dag.outputs)
    elif isinstance(dag, dict) and 'outputs' in dag:
        outputs = list(dag['outputs'])
    else:
        # Assume last vertex is output, or vertices with no children
        # For now, assume all vertices that are not parents of any other are outputs
        all_children = set()
        for v in vertices:
            for p in parents.get(v, []):
                all_children.add(p)
        outputs = [v for v in vertices if v not in all_children]
    
    return vertices, parents, outputs
=======
def _get_dag_info(dag):
    """Extract vertices, parents, outputs from dag object."""
    # Extract parents first to potentially derive vertices
    if hasattr(dag, 'parents'):
        parents = dict(dag.parents)
    elif isinstance(dag, dict) and 'parents' in dag:
        parents = dict(dag['parents'])
    else:
        parents = {}
    
    # Extract vertices
    if hasattr(dag, 'vertices'):
        vertices = list(dag.vertices)
    elif isinstance(dag, dict) and 'vertices' in dag:
        vertices = list(dag['vertices'])
    else:
        # Fallback: derive vertices from parents keys
        # Also include any vertices that appear as parents (though they should be keys)
        # In a standard DAG representation, keys of parents dict are the vertices.
        vertices = list(parents.keys())
        
        # If vertices is still empty, check if dag itself is a list of vertices?
        if not vertices and isinstance(dag, list):
            vertices = list(dag)
        elif not vertices and isinstance(dag, tuple):
            vertices = list(dag)
            
        if not vertices:
            raise ValueError("Cannot determine vertices")

    # Extract outputs
    if hasattr(dag, 'outputs'):
        outputs = list(dag.outputs)
    elif isinstance(dag, dict) and 'outputs' in dag:
        outputs = list(dag['outputs'])
    else:
        # Derive outputs: vertices that are not parents of any other vertex
        # i.e., sink nodes in the DAG
        parents_of = set()
        for v in vertices:
            for p in parents.get(v, []):
                parents_of.add(p)
        outputs = [v for v in vertices if v not in parents_of]
        
        # If no outputs found (e.g., single node), assume all vertices are outputs? 
        # Or just the last one in topo order? 
        # Standard definition: outputs are specified. If not, sinks are outputs.
        if not outputs and vertices:
            outputs = [vertices[-1]]
    
    return vertices, parents, outputs
>>>>>>> REPLACE

