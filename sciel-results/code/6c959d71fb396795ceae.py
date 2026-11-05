    def naive_schedule():
        ops = []
        # Load all inputs
        for v in inputs:
            ops.append(['load', v])
        # Process in topological order
        for v in topo_order:
            if v in inputs:
                continue
            # Check if all parents are in fast memory (we'll track this)
            # For naive: just compute and keep all
            ops.append(['compute', v])
        # Store all outputs
        for v in outputs:
            ops.append(['store', v])
        return ops
