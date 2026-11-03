<<<<<<< SEARCH
                            # Remove both
                            del terms[a] if a in terms else None
                            del terms[b] if b in terms else None
=======
                            # Remove both
                            if a in terms:
                                del terms[a]
                            if b in terms:
                                del terms[b]
>>>>>>> REPLACE
