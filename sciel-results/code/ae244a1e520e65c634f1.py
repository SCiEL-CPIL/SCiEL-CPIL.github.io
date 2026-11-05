<<<<<<< SEARCH
    n = len(dag)
    parents = dag
=======
    # Handle both list and dict formats for dag to prevent KeyError if indices are 1-based or dict-based
    if isinstance(dag, dict):
        try:
            n = max(dag.keys()) + 1
        except ValueError:
            n = 0
        parents = [dag.get(i, []) for i in range(n)]
    else:
        n = len(dag)
        parents = dag
>>>>>>> REPLACE
