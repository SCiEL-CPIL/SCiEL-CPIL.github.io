<<<<<<< SEARCH
        if len(code) < best_size or best_code is None:
            if time.time() < deadline - 10:
                code = local_search_improve(code, deadline - 5)
                code = reduce_code_fast(code)
=======
        if len(code) < best_size or best_code is None:
            if time.time() < deadline - 20:
                code = local_search_improve(code, deadline - 10)
                code = reduce_code_fast(code)
>>>>>>> REPLACE
