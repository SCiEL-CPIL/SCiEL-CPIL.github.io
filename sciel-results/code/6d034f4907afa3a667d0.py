        if v in outputs:
            if v not in fast_mem:
                # If it's an input, load it. If it's not an input, it should have been computed.
                # If it wasn't computed (error), we can't fix it easily, but inputs are the main case.
                if v in inputs:
                    operations.append(["load", v])
                    fast_mem.add(v)
                else:
                    # This should not happen in a valid DAG traversal if compute was called
                    # But if memory was tight and v was deleted? No, v is just computed.
                    pass
            operations.append(["store", v])
            fast_mem.remove(v)
        