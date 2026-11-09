        if i in set(inputs):
            if i not in in_fast:
                # It was stored, load it back
                ops.append(['load', get_vertex_id(i)])
                in_fast.add(i)
            # If it's an output, it stays.
            # If it's not an output, it might be stored later if space is tight.
            # But we don't compute it.
            pass
